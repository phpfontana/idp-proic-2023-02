from ultralytics import YOLO


def main():
    # Load a model pretrained model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data='../dataset/ProIC-1/data.yaml',    # path to data file
        epochs=300,                             # number of epochs
        batch=16,                               # batch size
        imgsz=640,                              # image size
        device="mps",                           # cuda device, i.e. 0 or 0,1,2,3 or cpu
        patience=100,                           # early stopping patience
        save=True,                              # save checkpoint every epoch
        save_period=5,                          # checkpoint save every period
        project='ProIC-01',
        name='yolov8n-01',
    )


if __name__ == '__main__':
    main()
