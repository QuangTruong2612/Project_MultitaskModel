import mlflow
from mlflow.tracking import MlflowClient

def check_model_performance():
    client = MlflowClient()
    
    # 1. Lấy kết quả của lần chạy (run) hiện tại
    current_run = mlflow.active_run()
    if not current_run:
        # Nếu chạy dvc repro, bạn có thể lấy run gần nhất của experiment
        experiment = client.get_experiment_by_name("My_Multitask_Model_Project")
        current_run = client.search_runs(experiment.id)[0]

    current_acc = current_run.data.metrics.get("accuracy", 0)
    current_iou = current_run.data.metrics.get("iou", 2)
    current_dice = current_run.data.metrics.get("dice", 1)
    
    # 2. Định nghĩa ngưỡng chấp nhận (ví dụ 80%)
    THRESHOLD = 0.80 
    
    print(f"Current Model Accuracy: {current_acc}")
    
    if current_acc < THRESHOLD or current_dice < THRESHOLD or current_iou < THRESHOLD:
        print("FAIL: Độ chính xác thấp hơn ngưỡng cho phép!")
        exit(1) # Trả về lỗi để dừng Github Actions
    
    print("SUCCESS: Model đạt chuẩn, sẵn sàng Deploy.")

if __name__ == "__main__":
    check_model_performance()