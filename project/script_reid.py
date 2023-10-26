import torchreid
import torch
import torch.nn as nn

def main():
    """Load data manager"""
    datamanager = torchreid.data.ImageDataManager(
        root="/home/sunho/hdd/sunho/personReID",
        sources="market1501",
        #targets="market1501", # Default is `sources`
        height=256,
        width=128,
        batch_size_train=32*2*torch.cuda.device_count(),
        batch_size_test=100,
        transforms=["random_flip", "random_crop", "color_jitter"]
    )

    """Build model, optimizer and lr_scheduler"""
    model = torchreid.models.build_model(
        name="osnet_ain_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=False
    )

    torchreid.utils.load_pretrained_weights(model=model, weight_path="/home/sunho/re-id/modelzoo/osnet_ain_x1_0_imagenet.pth")

    # 모델을 DataParallel로 래핑합니다.
    if torch.cuda.device_count() > 1:
        print(":::::::::::::::::Using", torch.cuda.device_count(), "GPUs!:::::::::::::::::")
        model = nn.DataParallel(model)

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    """Build engine"""
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    """Run training and test"""
    engine.run(
        save_dir="log/osnet_ain_x1_0/market1501/imagenet_cosine",
        max_epoch=60,
        eval_freq=20,
        print_freq=40,
        dist_metric='cosine',
        test_only=False #,fixbase_epoch=5, open_layers=['fc', 'classifier']
    )

if __name__ == '__main__':
    main()
