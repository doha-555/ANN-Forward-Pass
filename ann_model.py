import numpy as np

# 1. المدخلات (بناءً على المثال السابق)
inputs = np.array([0.05, 0.10])

# 2. تحديد الأوزان عشوائياً بين [-0.5, 0.5]
np.random.seed(42) # لضمان ثبات الأرقام عند كل تشغيل
weights_ih = np.random.uniform(-0.5, 0.5, (2, 2)) # أوزان الطبقة المخفية
weights_ho = np.random.uniform(-0.5, 0.5, (2, 2)) # أوزان طبقة المخرجات

# 3. تحديد الانحيازات (Biases) كما هو مطلوب
b1, b2 = 0.5, 0.7

# 4. دالة التنشيط tanh
def tanh_activation(x):
    return np.tanh(x)

# --- التمرير الأمامي (Forward Pass) ---
# حساب مخرجات الطبقة المخفية
net_h = np.dot(inputs, weights_ih) + b1
out_h = tanh_activation(net_h)

# حساب مخرجات الشبكة النهائية
net_o = np.dot(out_h, weights_ho) + b2
final_output = tanh_activation(net_o)

# 5. طباعة المخرجات
print("--- نتائج الشبكة العصبية (ANN Results) ---")
print(f"المخرجات (Final Outputs): {final_output}")