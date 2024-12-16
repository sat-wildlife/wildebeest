# ultralytics YOLO 🚀, AGPL-3.0 license

import platform
import threading
from pathlib import Path
from scipy.ndimage import zoom
import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox, classify_transforms
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from osgeo import gdal, ogr, osr
import os
import fnmatch

rsfilepath='D:/imagefile/'
rsshppath ='D:/outputfile/'

if not os.path.exists(rsshppath):
    os.makedirs(rsshppath)
    print("creat", rsshppath)
else:
    print("creat no need", rsshppath)

def percent_clip(image_data, lower_percent=0.5, upper_percent=99.5, out_min=0, out_max=255):
    if lower_percent >= upper_percent:
        raise ValueError("lower_percent should be less than upper_percent.")
    in_min = np.percentile(image_data, lower_percent)
    in_max = np.percentile(image_data, upper_percent)
    out_data = np.clip((image_data - in_min) / (in_max - in_min), 0, 1) * (out_max - out_min) + out_min
    return out_data.astype(np.uint8)

def linear_stretch(image_data, out_min=0, out_max=255):
    in_min = np.min(image_data)
    in_max = np.max(image_data)
    if in_min == in_max:
        return image_data
    else:
        out_data = (image_data - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
        return out_data.astype(np.uint8)

def read_image_in_windows(filename, window_size, overlap_control):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    bands_to_read = min(bands, 3)
    y = 0
    while y < height:
        x = 0
        while x < width:
            overlap = overlap_control['overlap']
            step_size = int(window_size * (1 - overlap))
            window_width = min(window_size, width - x)
            window_height = min(window_size, height - y)

            window_data = np.zeros((bands_to_read, window_height, window_width), dtype=np.uint8)

            for band in range(bands_to_read):
                gdal_band = dataset.GetRasterBand(band + 1)
                data = gdal_band.ReadAsArray(x, y, window_width, window_height)
                window_data[band, :window_height, :window_width] = data

            window_data = np.transpose(window_data, (1, 2, 0))
            #mirror_window_data = np.flip(window_data, axis=1)  # 沿着宽度维度反转，实现左右镜像
            top_left_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
            top_left_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]

            yield window_data,geotransform, projection, top_left_x, top_left_y

            x += step_size
        y += step_size


def read_image_in_windows_setoverlap(filename, window_size=336, overlap=0.5):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    bands_to_read = min(bands, 3)
    step_size = int(window_size * (1 - overlap))

    for y in range(0, height - step_size, step_size):
        for x in range(0, width - step_size, step_size):
            window_width = min(window_size, width - x)
            window_height = min(window_size, height - y)

            window_data = np.zeros((bands_to_read, window_size, window_size), dtype=np.uint8)
            for band in range(bands_to_read):
                gdal_band = dataset.GetRasterBand(band + 1)
                data = gdal_band.ReadAsArray(x, y, window_width, window_height)
                new_band = band if band < 3 else 0  # 简化了映射逻辑
                window_data[new_band, :window_height, :window_width] = data

            window_data = np.transpose(window_data, (1, 2, 0))
            top_left_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
            top_left_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]

            yield window_data, geotransform, projection, top_left_x, top_left_y

def read_image_in_windows_ori(filename, window_size=336):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    bands_to_read = min(bands, 3)

    for y in range(0, height, window_size):
        for x in range(0, width, window_size):
            window_width = min(window_size, width - x)
            window_height = min(window_size, height - y)
            window_data = np.zeros((bands_to_read, window_size, window_size), dtype=np.uint8)

            channel_mapping = {
                0: 0,
                1: 1,
                2: 2,
            }
            for band in range(bands_to_read):
                gdal_band = dataset.GetRasterBand(band + 1)
                data = gdal_band.ReadAsArray(x, y, window_width, window_height)
                new_band = channel_mapping.get(band, band)
                window_data[new_band, :window_height, :window_width] = data
            window_data = np.transpose(window_data, (1, 2, 0))

            top_left_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
            top_left_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]

            yield window_data, geotransform, projection, top_left_x, top_left_y


class BasePredictor:
    """
    BasePredictor.

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer, self.vid_frame = None, None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im, *args, **kwargs):
        """Runs inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        letterbox = LetterBox(self.imgsz, auto=same_shapes and self.model.pt, stride=self.model.stride)
        return [letterbox(image=x) for x in im]

    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            log_string += f"{idx}: "
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, "frame", 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / "labels" / p.stem) + ("" if self.dataset.mode == "image" else f"_{frame}")
        log_string += "%gx%g " % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = {
                "line_width": self.args.line_width,
                "boxes": self.args.show_boxes,
                "conf": self.args.show_conf,
                "labels": self.args.show_labels,
            }
            if not self.args.retina_masks:
                plot_args["im_gpu"] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        # Write
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(
                save_dir=self.save_dir / "crops",
                file_name=self.data_path.stem + ("" if self.dataset.mode == "image" else f"_{frame}"),
            )

        return log_string

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions for an image and returns them."""
        return preds

    def __call__(self, source=None, model=None, stream=False, *args, **kwargs):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for CLI prediction.

        It uses always generator as outputs as not required by CLI mode.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # noqa, running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = (
            getattr(
                self.model.model,
                "transforms",
                classify_transforms(self.imgsz[0], crop_fraction=self.args.crop_fraction),
            )
            if self.args.task == "classify"
            else None
        )
        self.dataset = load_inference_source(
            source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride, buffer=self.args.stream_buffer
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
                self.dataset.mode == "stream"  # streams
                or len(self.dataset) > 1000  # images
                or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path = [None] * self.dataset.bs
        self.vid_writer = [None] * self.dataset.bs
        self.vid_frame = [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):

        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                self.batch = batch
                path, im0s, vid_cap, s = batch

                ogr.RegisterAll()
                img_list = fnmatch.filter(os.listdir(rsfilepath), '*.tif')

                for imgktif in img_list:
                    print(imgktif)
                    imgkshp = imgktif[:-4] + '.shp'
                    rsfilename = os.path.join(rsfilepath, imgktif)
                    rsshpname = os.path.join(rsshppath, imgkshp)
                    oDriver = ogr.GetDriverByName('ESRI Shapefile')
                    oDS = oDriver.CreateDataSource(rsshpname)
                    pre_dataset = gdal.Open(rsfilename, gdal.GA_ReadOnly)
                    pre_geoproj = pre_dataset.GetProjection()
                    srs = osr.SpatialReference(wkt=pre_geoproj)
                    oLayer = oDS.CreateLayer("polygon", srs, ogr.wkbPolygon)
                    oFieldID = ogr.FieldDefn("Conf", ogr.OFTReal)
                    oLayer.CreateField(oFieldID, 1)
                    oDefn = oLayer.GetLayerDefn()

                    overlap_control = {'overlap': 0.5}

                    for window_i, geot, geoproj, top_left_x, top_left_y in read_image_in_windows(rsfilename,
                                                                                                 window_size=336,
                                                                                                 overlap_control=overlap_control):

                        window = []
                        window.append(window_i)
                        im_window = self.preprocess(window)
                        im_preds = self.inference(im_window, *args, **kwargs)
                        if self.args.embed:
                            yield from [im_preds] if isinstance(im_preds,
                                                                torch.Tensor) else im_preds  # yield embedding tensors
                            continue
                        window_results = self.postprocess(im_preds, im_window, window)
                        res_boxes = window_results[0].boxes
                        res_conf = res_boxes.conf.to('cpu').numpy()
                        res_xyxy = res_boxes.xyxy
                        res_xyxy = res_xyxy.to('cpu')
                        res_xyxy_np = res_xyxy.numpy()

                        if len(res_xyxy_np)>30:
                            overlap_control['overlap'] = 0.8
                        else:
                            overlap_control['overlap'] = 0.5

                        if len(res_xyxy_np) > 1:

                            for ni in range(0, len(res_conf)):
                                xmin = float(res_xyxy_np[ni][0])
                                ymin = float(res_xyxy_np[ni][1])
                                xmax = float(res_xyxy_np[ni][2])
                                ymax = float(res_xyxy_np[ni][3])
                                edge_margin=2

                                kk=0

                                bbox_width = xmax - xmin
                                bbox_height = ymax - ymin

                                if (xmin <= edge_margin or ymin <= edge_margin or
                                        xmax >= 336 - edge_margin or ymax >= 336 - edge_margin or bbox_width>10 or bbox_height>10):
                                    kk=1
                                    continue

                                geoxmin = top_left_x + xmin * geot[1] + ymin * geot[2]
                                geoymin = top_left_y + xmin * geot[4] + ymin * geot[5]
                                geoxmax = top_left_x + xmax * geot[1] + ymax * geot[2]
                                geoymax = top_left_y + xmax * geot[4] + ymax * geot[5]

                                if kk==1:
                                    print('error...')

                                ring = ogr.Geometry(ogr.wkbLinearRing)
                                ring.AddPoint(geoxmin, geoymin)
                                ring.AddPoint(geoxmax, geoymin)
                                ring.AddPoint(geoxmax, geoymax)
                                ring.AddPoint(geoxmin, geoymax)
                                ring.CloseRings()
                                poly = ogr.Geometry(ogr.wkbPolygon)
                                poly.AddGeometry(ring)
                                outfeat = ogr.Feature(oDefn)
                                outfeat.SetField(0, str(res_conf[ni]))
                                outfeat.SetGeometry(poly)
                                oLayer.CreateFeature(outfeat)

                    oDS.Destroy()

                self.run_callbacks("on_predict_postprocess_end")
                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))
                    if self.args.save or self.args.save_txt:
                        self.results[i].save_dir = self.save_dir.__str__()
                    if self.args.show and self.plotted_img is not None:
                        self.show(p)
                    if self.args.save and self.plotted_img is not None:
                        self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

                # Print time (inference-only)
                if self.args.verbose:
                    LOGGER.info(f"{s}{profilers[1].dt * 1E3:.1f}ms")

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(1, 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")

    def stream_inference_ori(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if not self.model:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                self.batch = batch
                path, im0s, vid_cap, s = batch
                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue
                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)

                self.run_callbacks("on_predict_postprocess_end")
                # Visualize, save, write results
                n = len(im0s)
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "preprocess": profilers[0].dt * 1e3 / n,
                        "inference": profilers[1].dt * 1e3 / n,
                        "postprocess": profilers[2].dt * 1e3 / n,
                    }
                    p, im0 = path[i], None if self.source_type.tensor else im0s[i].copy()
                    p = Path(p)

                    if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                        s += self.write_results(i, self.results, (p, im, im0))
                    if self.args.save or self.args.save_txt:
                        self.results[i].save_dir = self.save_dir.__str__()
                    if self.args.show and self.plotted_img is not None:
                        self.show(p)
                    if self.args.save and self.plotted_img is not None:
                        self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

                # Print time (inference-only)
                if self.args.verbose:
                    LOGGER.info(f"{s}{profilers[1].dt * 1E3:.1f}ms")

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(1, 3, *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = AutoBackend(
            model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        im0 = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(500 if self.batch[3].startswith("image") else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        im0 = self.plotted_img
        # Save imgs
        if self.dataset.mode == "image":
            cv2.imwrite(save_path, im0)
        else:  # 'video' or 'stream'
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                    self.vid_frame[idx] = 0
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))  # integer required, floats produce error in MP4 codec
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[idx] = cv2.VideoWriter(
                    str(Path(save_path).with_suffix(suffix)), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
                )
            # Write video
            self.vid_writer[idx].write(im0)

            # Write frame
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}{self.vid_frame[idx]}.jpg", im0)
                self.vid_frame[idx] += 1

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)
