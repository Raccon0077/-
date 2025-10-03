import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# 1. Генерация данных
np.random.seed(42)  # для воспроизводимости результатов

# Генерируем x1 и x2 с использованием linspace
x1 = np.linspace(-10, 10, 500)
x2 = np.linspace(-8, 8, 500)

# Создаем DataFrame
df = pd.DataFrame({
    'x1': x1,
    'x2': x2
})

# Вычисляем y по функции: y = -2x₁² + x₁x₂ + 4x₂² + 5
df['y'] = -2*df['x1']**2 + df['x1']*df['x2'] + 4*df['x2']**2 + 5

# 2. Сохраняем в CSV файл
df.to_csv('generated_data.csv', index=False)
print("Файл 'generated_data.csv' создан успешно!")
print(f"Размер данных: {df.shape}")

# 3. Выводим статистику по столбцам
print("\nСтатистика по столбцам:")
for col in df.columns:
    print(f"{col}: среднее = {df[col].mean():.2f}, минимум = {df[col].min():.2f}, максимум = {df[col].max():.2f}")

# 4. Строим графики с использованием matplotlib
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# График 1: y от x1 (x2 = константа)
x2_const = df['x2'].mean()  # берем среднее значение x2 как константу
y_vs_x1 = -2*df['x1']**2 + df['x1']*x2_const + 4*x2_const**2 + 5

ax1.plot(df['x1'], y_vs_x1, 'b-', linewidth=2)
ax1.set_xlabel('x1')
ax1.set_ylabel('y')
ax1.set_title(f'График y от x1 (x2 = {x2_const:.2f})')
ax1.grid(True, alpha=0.3)

# График 2: y от x2 (x1 = константа) с точками
x1_const = df['x1'].mean()  # берем среднее значение x1 как константу
y_vs_x2 = -2*x1_const**2 + x1_const*df['x2'] + 4*df['x2']**2 + 5

ax2.scatter(df['x2'], y_vs_x2, c='red', s=10, alpha=0.6)
ax2.plot(df['x2'], y_vs_x2, 'r-', alpha=0.7)
ax2.set_xlabel('x2')
ax2.set_ylabel('y')
ax2.set_title(f'График y от x2 (x1 = {x1_const:.2f})')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Сохраняем строки, где x1 < среднее(x1) ИЛИ x2 < среднее(x2)
mean_x1 = df['x1'].mean()
mean_x2 = df['x2'].mean()

filtered_df = df[(df['x1'] < mean_x1) | (df['x2'] < mean_x2)]
filtered_df.to_csv('filtered_data.csv', index=False)
print(f"\nОтфильтрованные данные сохранены в 'filtered_data.csv'")
print(f"Исходные данные: {len(df)} строк")
print(f"Отфильтрованные данные: {len(filtered_df)} строк")

# 6. Строим 3D график функции
fig = plt.figure(figsize=(12, 9))
ax = plt.axes(projection='3d')

# Создаем сетку для 3D графика
X1, X2 = np.meshgrid(np.linspace(-10, 10, 50),
                     np.linspace(-8, 8, 50))
Y = -2*X1**2 + X1*X2 + 4*X2**2 + 5

# Рисуем поверхность
surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8,
                      linewidth=0, antialiased=True)

# Добавляем цветовую шкалу
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D график функции: y = -2x₁² + x₁x₂ + 4x₂² + 5')

plt.tight_layout()
plt.show()

# Дополнительная информация
print("\n" + "="*50)
print("ОТЧЕТ О ВЫПОЛНЕНИИ:")
print("="*50)
print(f"1. Сгенерировано {len(df)} строк данных")
print(f"2. Создан файл: generated_data.csv")
print(f"3. Построены 2D графики функции")
print(f"4. Рассчитана статистика по всем столбцам")
print(f"5. Создан файл с отфильтрованными данными: filtered_data.csv")
print(f"6. Построен 3D график функции в отдельном окне")
print(f"7. Функция: y = -2x₁² + x₁x₂ + 4x₂² + 5")