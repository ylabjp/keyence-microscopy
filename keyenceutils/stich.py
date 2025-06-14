import glob
import re
import tifffile as tiff
import numpy as np
import os
import pandas as pd
from keyenceutils.metainfo import ImageMetadata
from skimage.registration import phase_cross_correlation
from scipy.ndimage import distance_transform_edt
from collections import deque

class Plane:
    def __init__(self, meta_info:pd.DataFrame,folder_path):

        self.__meta_info = meta_info    # type: pd.DataFrame
        assert self.__meta_info["Z"].nunique() == 1, "All images must be in the same Z plane."
        assert self.__meta_info["CH"].nunique() == 1, "All images must have the same CH value."
        
        self.images = {
            row.fname: tiff.imread(os.path.join(folder_path, row.fname)) 
            for _, row in self.__meta_info.iterrows()
        }
        pairwise_shifts = self._calculate_all_pairwise_shifts()
        final_positions = self._determine_final_positions(pairwise_shifts)
        self.canvas_array = self._render_final_image(final_positions)

    def _calculate_all_pairwise_shifts(self):
        """
        ステップ2: 隣接するタイル間の高精度なピクセル単位のズレを計算する
        """
        print("ステップ2: 隣接タイル間のピクセルシフト計算を開始...")
        shifts = {}
        # 効率化のため、タイルを位置でソート
        sorted_tiles = self.__meta_info.sort_values(by=['Y_relative', 'X_relative'])

        for i, tile1 in sorted_tiles.iterrows():
            # 右隣のタイルを探す
            right_neighbors = sorted_tiles[
                (abs(tile1.Y_relative - sorted_tiles.Y_relative) < tile1.H / 2) &
                (sorted_tiles.X_relative > tile1.X_relative)
            ]
            if not right_neighbors.empty:
                tile2 = right_neighbors.iloc[0]
                overlap_w = int((tile1.X_relative + tile1.W) - tile2.X_relative)
                if overlap_w > 10: # 最小オーバーラップ幅
                    img1_overlap = self.images[tile1.fname][:, -overlap_w:]
                    img2_overlap = self.images[tile2.fname][:, :overlap_w]
                    shift, _, _ = phase_cross_correlation(img1_overlap, img2_overlap, upsample_factor=10)
                    shifts[(tile1.fname, tile2.fname)] = shift # (dy, dx)
            
            # 下隣のタイルを探す
            down_neighbors = sorted_tiles[
                (abs(tile1.X_relative - sorted_tiles.X_relative) < tile1.W / 2) &
                (sorted_tiles.Y_relative > tile1.Y_relative)
            ]
            if not down_neighbors.empty:
                tile2 = down_neighbors.iloc[0]
                overlap_h = int((tile1.Y_relative + tile1.H) - tile2.Y_relative)
                if overlap_h > 10:
                    img1_overlap = self.images[tile1.fname][-overlap_h:, :]
                    img2_overlap = self.images[tile2.fname][:overlap_h, :]
                    shift, _, _ = phase_cross_correlation(img1_overlap, img2_overlap, upsample_factor=10)
                    shifts[(tile1.fname, tile2.fname)] = shift

        print(f"{len(shifts)} 個の隣接ペアについてシフトを計算しました。")
        return shifts

    def _determine_final_positions(self, pairwise_shifts):
        """
        ステップ3: 計算したズレを元に、全タイルの最終的な絶対座標を決定する
        """
        print("ステップ3: 全タイルの最終座標の計算を開始...")
        # 最初のタイルをアンカー（基準）とする
        anchor_idx = self.__meta_info.sort_values(by=['Y_relative', 'X_relative']).index[0]
        
        final_positions = {anchor_idx: (self.__meta_info.loc[anchor_idx, "Y_relative"], 
                                        self.__meta_info.loc[anchor_idx, "X_relative"])}
        
        # 幅優先探索で位置を伝播させる
        q = deque([anchor_idx])
        processed = {anchor_idx}

        while q:
            current_idx = q.popleft()
            
            # 隣接ペアを見つけてキューに追加
            for (idx1, idx2), shift in pairwise_shifts.items():
                neighbor_idx = None
                base_idx = None
                if idx1 == current_idx and idx2 not in processed:
                    neighbor_idx, base_idx = idx2, idx1
                elif idx2 == current_idx and idx1 not in processed:
                    neighbor_idx, base_idx = idx1, idx2
                
                if neighbor_idx is not None:
                    base_tile = self.__meta_info.loc[base_idx]
                    neighbor_tile = self.__meta_info.loc[neighbor_idx]
                    
                    # 基準タイルの確定済み座標
                    base_pos_y, base_pos_x = final_positions[base_idx]
                    
                    # メタデータ上の理論的な相対位置
                    initial_offset_y = neighbor_tile.Y_relative - base_tile.Y_relative
                    initial_offset_x = neighbor_tile.X_relative - base_tile.X_relative

                    # 位相限定相関法で計算したズレ
                    dy, dx = shift if base_idx == idx1 else -shift

                    # 最終座標を計算
                    final_y = base_pos_y + initial_offset_y + dy
                    final_x = base_pos_x + initial_offset_x + dx

                    final_positions[neighbor_idx] = (final_y, final_x)
                    processed.add(neighbor_idx)
                    q.append(neighbor_idx)

        print("全タイルの最終座標を決定しました。")
        return final_positions
    

    def _render_final_image(self, final_positions):
        """
        ステップ4: 補正された最終座標を元に、ブレンディングしながら画像をレンダリングする
        """
        print("ステップ4: 最終画像のレンダリングを開始...")
        tile_H, tile_W = self.__meta_info.loc[0, "H"], self.__meta_info.loc[0, "W"]
        
        # 最終的なキャンバスサイズを計算
        min_y = min(pos[0] for pos in final_positions.values())
        min_x = min(pos[1] for pos in final_positions.values())
        max_y = max(pos[0] for pos in final_positions.values())
        max_x = max(pos[1] for pos in final_positions.values())
        
        canvas_H = int(np.ceil(max_y + tile_H - min_y))
        canvas_W = int(np.ceil(max_x + tile_W - min_x))


        canvas = np.zeros((canvas_H, canvas_W), dtype=np.float32)
        weight_canvas = np.zeros_like(canvas)
        
        # ブレンディング用の重みマップ (全タイルで共通)
        weight_map = distance_transform_edt(np.ones((tile_H, tile_W), dtype=np.float32))
        weight_map /= np.max(weight_map)


        for idx, row in self.__meta_info.iterrows():
            if idx not in final_positions: continue
            
            img = self.images[row.fname].astype(np.float32)
            # 元のコードにあった反転処理を適用
            img = np.flipud(np.fliplr(img))

            y_start_f, x_start_f = final_positions[idx]
            # 全体がマイナスにならないようにオフセット
            y_start = int(round(y_start_f - min_y))
            x_start = int(round(x_start_f - min_x))

            if y_start < 0 or x_start < 0 or (y_start + tile_H > canvas_H) or (x_start + tile_W > canvas_W):
                continue

            # 線形ブレンディング
            roi = canvas[y_start : y_start + tile_H, x_start : x_start + tile_W]
            roi += img * weight_map
            weight_canvas[y_start : y_start + tile_H, x_start : x_start + tile_W] += weight_map

        # 正規化
        canvas /= np.maximum(weight_canvas, 1e-6)

        
        # (Z, C, Y, X) の形にスタックする
        return canvas

class StichedImage:
    """
    StichedImage is a utility class for processing and stitching microscopy images 
    stored in a folder. It handles multiple channels, extracts metadata, and 
    creates a stitched image canvas. The class also provides functionality to save 
    the stitched image with ImageJ-compatible metadata.
    Attributes:
        __canvas_array (np.ndarray): A 3D array representing the stitched image 
            canvas, where each channel is stored as a separate layer.
        __meta_info (pd.DataFrame): A DataFrame containing metadata for all images, 
            including their positions, dimensions, and channels.
        __channels (list): A list of unique channel identifiers found in the images.
        __nm_per_pixel_values (float): The nanometers per pixel value extracted 
            from the metadata of the first image.
    Methods:
        __init__(folder_path):
            Initializes the StichedImage object by processing the images in the 
            specified folder (e.g. XY01), extracting metadata, and creating the stitched canvas.
        get_meta_info():
            Returns the metadata DataFrame containing information about the images.
        get_channels():
            Returns the list of unique channel identifiers.
        save(output_path):
            Saves the stitched image as a TIFF file with ImageJ-compatible metadata.
    """

    def __init__(self, folder_path, user_metadata={}):
        self.__canvas_array = None  # type: np.ndarray
        self.__meta_info = None  # type: pd.DataFrame
        self.__channels = None  # type: list
        self.__user_metadata = user_metadata  # type: dict

        print(f"Processing folder: {folder_path}")

        # find all TIFF files in specified folder
        all_tif_files = sorted(
            glob.glob(os.path.join(folder_path, "Image_*.tif"))
        )
        # Filter out TIFF files that contain "Overlay" in their filename
        all_tif_files = [os.path.basename(
            f) for f in all_tif_files if "Overlay" not in os.path.basename(f)]

        # Identify unique channels
        self.__channels = sorted(
            {re.search(r'CH\d+', f).group()
             for f in all_tif_files if re.search(r'CH\d+', f)}
        )

        assert all_tif_files is not None, "No TIFF files found in the specified folder."
        assert len(
            self.__channels) > 0, "No channels found in the specified folder."
        assert len(
            all_tif_files) > 0, "No TIFF files found in the specified folder."

        # check for Z stack images
        zstack_mode = all(
            [re.search(r'_Z\d+_', f) is not None for f in all_tif_files])

        meta_info = []
        z_max = 0

        for c_idx, channel in enumerate(self.__channels):
            print(f"Processing channel: {channel} in folder {folder_path}")
            tif_files = sorted(
                [f for f in all_tif_files if f.endswith(f"{channel}.tif")])

            d = pd.DataFrame(
                list(map(lambda tif: ImageMetadata(
                    os.path.join(
                        folder_path, tif)).get_dict(), tif_files)
                     )
            )
            d["CH_idx"] = c_idx
            d["CH"] = channel
            d["fname"] = tif_files
            meta_info.append(d)

            if zstack_mode:
                d["Z"] = d["fname"].apply(lambda x: int(re.search(r'_Z(\d+)_', x).group(1))
                                          if re.search(r'_Z(\d+)_', x) else 0)
                z_max = max(z_max, d["Z"].max()+1)

            else:
                d["Z"] = 0
                z_max = 1

            meta_info.append(d)

        meta_info = pd.concat(meta_info)
        meta_info["X_relative"] = meta_info["X"]-meta_info["X"].min()
        meta_info["Y_relative"] = meta_info["Y"]-meta_info["Y"].min()

        # Calculate the position relative to the bottom-right corner of the canvas
        canvas_width = meta_info["X_relative"].max() + meta_info["W"].max()
        canvas_height = meta_info["Y_relative"].max() + meta_info["H"].max()

        # create a canvas with zcyx shape
        canvas_array = np.zeros(
            (
                z_max,
                meta_info["CH_idx"].nunique(),
                canvas_height,
                canvas_width
            ),
            dtype=np.uint16
        )  # Merged array for the canvas

        for label, group in meta_info.groupby(["Z","CH_idx"]):

            img = Plane(group, folder_path).canvas_array[0]

            selection = (
                label[0],
                label[1],
            )
            canvas_array[selection]=img


        self.__canvas_array = canvas_array
        self.__meta_info = meta_info


    def get_meta_info(self):
        return self.__meta_info

    def get_channels(self):
        return self.__channels

    def save(self, output_path):
        # Save the stitched image as a TIFF file
        # Save the stitched image as a TIFF file with ImageJ compatible metadata
        # https://imagej.net/ij/plugins/metadata/MetaData.pdf

        assert self.__meta_info["umPerPixel"].nunique(
        ) == 1, "All images must have the same umPerPixel value."
        assert self.__meta_info["LensName"].nunique(
        ) == 1, "All images must have the same LensName."
        # Exposure time may vary among channels
        # assert self.__meta_info["ExposureTimeInS"].nunique(
        # ) == 1, "All images must have the same ExposureTimeInS."
        exp_str = self.__meta_info.groupby(
            "CH")["ExposureTimeInS"].first().to_string()

        res = 1.0 / self.__meta_info["umPerPixel"].values[0]

        p = {
            "Lens": self.__meta_info["LensName"].values[0],
            "ExposureTime(s)": exp_str,
            "Sectioning": self.__meta_info["Sectioning"].values[0]
        } | self.__user_metadata

        metadata = {
            'Properties': p,
            'axes': 'ZCYX',  # ImageJ is only compatible with TZCYXS order
            'hyperstack': True,
            'mode': 'composite',
            'spacing': res,
            'unit': 'um',
        }

        tiff.imwrite(
            output_path,
            self.__canvas_array,
            photometric='minisblack',
            imagej=True,
            resolution=(     # Number of pixels per `resolutionunit` in X and Y directions
                res,
                res,
            ),
            resolutionunit=tiff.RESUNIT.MICROMETER,
            metadata=metadata,
        )
        pass
