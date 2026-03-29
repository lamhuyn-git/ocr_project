from paddlex import create_model

# Load model recognition tiếng Việt (PP-OCRv5 server - độ chính xác cao nhất)
model = create_model("PP-OCRv5_server_rec")

# Fine-tune
model.train(
    dataset_type      = "MSTextRecDataset",
    dataset_root_path = "./image_train",          # thư mục gốc chứa crop_img/ + train/ + val/
    train_annot_path  = "train/rec_gt_train.txt", # đường dẫn tương đối so với dataset_root_path
    val_annot_path    = "val/rec_gt_val.txt",
    char_dict_path = "./dict_vi.txt",
    # Hyperparameters
    epochs_iters  = 50,       # số epoch — tăng lên 80-100 nếu val_acc chưa đạt 0.90
    batch_size    = 16,       # phù hợp với RAM máy Mac; giảm xuống 8 nếu bị lỗi OOM
    learning_rate = 0.0001,   # learning rate nhỏ vì đang fine-tune (không train từ đầu)

    # Lưu model vào thư mục này
    output = "./output_rec",
)

print("\n Fine-tune successful! Model saved at: ./output_rec/best_model")
print("   Next step: update src/ocr_engine.py to use the new model.")
