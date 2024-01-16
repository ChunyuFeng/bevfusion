import os
from typing import Any, Dict, Tuple

import cv2
import mmcv
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


# tools/create_data.py 并没有处理map，只生成了obstacle的gt
# 所有用到local map的地方，都是根据cfg文件，从LoadBEVSegmentation中获取的
@PIPELINES.register_module()
class LoadBEVSegmentation:
    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # The following code is used to calculate the transformation matrix from LiDAR coordinates to global coordinates.
        # First, we retrieve the transformation matrix from LiDAR coordinates to augmented point cloud coordinates.
        lidar2point = data["lidar_aug_matrix"]
        # We then calculate the inverse of this matrix to get the transformation from the augmented point cloud coordinates back to LiDAR coordinates.
        point2lidar = np.linalg.inv(lidar2point)
        # Next, we retrieve the transformation matrix from LiDAR coordinates to ego vehicle coordinates.
        lidar2ego = data["lidar2ego"]
        # We also retrieve the transformation matrix from ego vehicle coordinates to global coordinates.
        ego2global = data["ego2global"]
        # Finally, we calculate the transformation matrix from LiDAR coordinates to global coordinates by multiplying the three transformation matrices together.
        # The order of multiplication is important here because matrix multiplication is not commutative.
        lidar2global = ego2global @ lidar2ego @ point2lidar

        # 以下代码用于从LiDAR坐标到全局坐标的转换矩阵中提取LiDAR传感器在全局坐标系统中的位置。
        # 位置表示为2D点（x，y）。
        map_pose = lidar2global[:2, 3]
        # 构造LiDAR传感器周围的局部地图的边界框。
        # 边界框表示为元组（x，y，width，height），
        # 其中（x，y）是LiDAR传感器的位置，width和height是局部地图的尺寸。
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        # 以下代码用于计算从LiDAR坐标到全局坐标的转换矩阵中的旋转部分，并计算出偏航角。
        # 首先，我们获取从LiDAR坐标到全局坐标的转换矩阵的旋转部分。
        rotation = lidar2global[:3, :3]
        # 然后，我们将旋转矩阵与向量[1, 0, 0]进行点积运算。
        v = np.dot(rotation, np.array([1, 0, 0]))
        # 接着，我们使用arctan2函数计算出偏航角。
        yaw = np.arctan2(v[1], v[0])
        # 最后，我们将偏航角从弧度转换为角度。
        patch_angle = yaw / np.pi * 180

        # 以下代码用于从局部地图中提取语义分割标签。
        # 我们首先定义一个字典，将语义分割标签的名称映射到地图中的图层名称。
        mappings = {}
        # 我们将“drivable_area”映射到“road_segment”和“lane”。
        # 我们将“divider”映射到“road_divider”和“lane_divider”。
        # 我们将其他语义分割标签映射到它们自己。
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # if "gt_bboxes_3d" in data and hasattr(data["gt_bboxes_3d"], 'corners') and data["gt_bboxes_3d"].corners is not None and len(data["gt_bboxes_3d"].corners) != 0:
        if "gt_bboxes_3d" in data and len(data["gt_bboxes_3d"]) > 0 and hasattr(data["gt_bboxes_3d"], 'corners'):
            if data["gt_bboxes_3d"].corners is not None and len(data["gt_bboxes_3d"].corners) != 0:
                # 以下代码用于将BEV obstacles的信息加入地图掩膜中。
                obs_bev_coords = data["gt_bboxes_3d"].corners[:, [0, 3, 7, 4], :2]
                # 在map里创建obstacle层不合理，所以这里直接把obstacle的mask加到drivable_area里
                # 创建same size的全0 矩阵
                obs_mask = np.zeros((self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
                canvas_center = (obs_mask.shape[1] // 2, obs_mask.shape[0] // 2)
                for index in range(obs_bev_coords.shape[0]):
                    # 画出每个obstacle的mask
                    bbox = obs_bev_coords[index]
                    bbox = bbox.numpy().astype(np.int32)
                    obs_mask = cv2.fillPoly(obs_mask, [bbox], 1, offset=canvas_center)
                # 水平翻转
                obs_mask = np.flip(obs_mask, axis=0)
                # 顺时针旋转90度，与map的坐标系对齐
                # 获取图像的形状
                h, w = obs_mask.shape
                # 计算旋转中心
                center = (w / 2, h / 2)
                # 获取旋转矩阵
                M = cv2.getRotationMatrix2D(center, -90, 1)
                # 应用旋转矩阵
                obs_mask = cv2.warpAffine(obs_mask.astype(np.uint8), M, (w, h))
            else:
                obs_mask = obs_mask = np.zeros((self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
        else:
            obs_mask = obs_mask = np.zeros((self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)

        obs_mask = obs_mask.astype(np.bool)

        # 以下代码用于从数据中获取地图位置，并使用NuScenesMap的get_map_mask方法获取地图掩膜。
        # 地图位置由"data"字典中的"location"键获取。
        location = data["location"]
        # get_map_mask方法需要以下参数：
        # - patch_box：LiDAR传感器周围的局部地图的边界框。
        # - patch_angle：局部地图的偏航角。
        # - layer_names：需要获取的地图层的名称列表。
        # - canvas_size：画布大小，即输出地图掩膜的大小。
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # 由于地图掩膜的形状可能与预期的不同，因此我们需要对其进行转置以获得正确的形状。
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        # 最后，我们将地图掩膜的数据类型转换为布尔型，因为掩膜是二进制的。
        masks = masks.astype(np.bool)

        # 以下代码用于生成地图掩膜的标签。
        # 首先，我们获取类别的数量，并为每个类别创建一个全零的标签矩阵。
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        # 然后，我们遍历每个类别。
        for k, name in enumerate(self.classes):
            # 对于每个类别，我们遍历其对应的地图层。
            for layer_name in mappings[name]:
                # 我们找到当前地图层在所有地图层中的索引。
                index = layer_names.index(layer_name)
                # 然后，我们将当前地图层的掩膜应用到对应类别的标签矩阵上。
                # 这样，标签矩阵中的每个像素值将表示该像素是否属于当前类别。
                if layer_name == "drivable_area" and obs_mask is not None:
                        overlap = np.logical_and(obs_mask, masks[index])
                        drivable_area = np.logical_and(masks[index], np.logical_not(overlap))
                        labels[k, drivable_area] = 1
                else:
                    labels[k, masks[index]] = 1


        # labels[0, masks[5]] = 0

        # overlap = np.logical_and(obs_mask, masks[5])
        # labels[0, overlap] = 0

        data["gt_masks_bev"] = labels
        return data


@PIPELINES.register_module()
class LoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results
