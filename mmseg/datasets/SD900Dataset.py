from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SD900Dataset(BaseSegDataset):
    METAINFO = {
        # rolled - in scale(RS), patches(Pa), crazing(Cr), pitted surface(PS), inclusion(In) and scratches(Sc)
        'classes': ['background',
                    'In', 'Pa', 'Sc', ],
        'palette': [[0, 0, 0],
                    [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    }

    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 img_suffix='.bmp',
                 seg_map_suffix='.png',  # 标注mask图像的格式
                 reduce_zero_label=False,  # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
