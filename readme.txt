# hramr-matting-main-project-apply
图像前景分割算法应用

使用算法的infer.py  处理图像的alpha通道图像
使用imageKou-jingxihua 进行图像前景抠图处理

infer1.py 为原创作者代码，有bug，进行了修复

带有Transparent-460数据集，composited_images文件夹下为原始图像，trimap_copy文件夹下为三分图像
HRAMR-Matting-MDRDLoss文件夹下的图像为算法处理后的alpha通道图像

output_results 文件夹为imagekou-jingxihua 生成的图像

除了代码文件做了修改，文件目录也相应做了修改，该工程下载下来，在vs中可以直接运行，使用时，运行infer.py 或者 imageKou-jingxihua.py 文件夹为imagekou
该算法中用到的best_model.pth文件因为超出了500 MB 没办法上传，使用时需从github https://github.com/yexianmin/HRAMR-Matting 上找到 Models的链接进行下载，
链接为google link