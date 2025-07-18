import numpy as np

# Define the confusion matrices
cmats = [
    np.array([[1340, 61], 
              [141, 971]]),   # CNN
    np.array([[1385, 16],
              [43, 1069]]), #ConvNext
    np.array([[1343, 58],
              [217, 895]]), #VGG16
    np.array([[1359, 42],
              [247, 865]]), #MobileNetV2
    np.array([[1368, 33],
              [190, 922]]), #RESNet
    np.array([[1362, 39],
              [187, 925]]) #EffectLiteNet
]

def tinh_mdr_fdr(cmats):
    results = []
    for idx, cm in enumerate(cmats):
        total = np.sum(cm)
        sai = np.sum(cm) - np.trace(cm)
        false_positive = cm[0,1]  # Dự đoán sai là false positive
        total_sai = sai
        total_fp = false_positive

        # Tính MDR
        mdr = sai / total if total > 0 else 0

        # Tính FDR
        fdr = false_positive / sai if sai > 0 else 0

        results.append({
            'model_index': idx + 1,
            'MDR': mdr,
            'FDR': fdr
        })
    return results

# calculate MDR and FDR 
ket_qua = tinh_mdr_fdr(cmats)

# print results
for res in ket_qua:
    print(f"Model {res['model_index']}: MDR = {res['MDR']:.4f}, FDR = {res['FDR']:.4f}")