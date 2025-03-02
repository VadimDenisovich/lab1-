from scipy import stats 
import numpy as np
import math 
import pandas as pd 
import matplotlib.pyplot as plt

SEED = 3 
SIZE_100 = 100
SIZE_1000 = 1000 

# .rvs() — «random variates» — случайные величины. Генерирует случайные выборки 
# из указанного распределения вероятностей. 
# loc — нижняя граница распределения (минимальное значение, которое может принять СВ)
# scale — ширина интервала распределения. (В таком случае loc + scale = 15, это верх. граница распределения)
#
# Равн. распределение на интервале [5, 15]
uniform_100 = stats.uniform.rvs(loc=5, scale=10, size=SIZE_100, random_state=SEED)

# Распределение Бернулли, вероятность p = 0.7 
bernoulli_100 = stats.bernoulli.rvs(p=0.7, size=SIZE_100, random_state=SEED)

# Биноминальное распределение с n=20 (число испытаний), p=0.6 (вероятность успеха в каждом испытании) 
binominal_100 = stats.binom.rvs(n=20, p=0.6, size=SIZE_100, random_state=SEED)

# Нормальное распределение с параметрами mu=10, sigma=2, 
# где mu — матожидание, а sigma^2 — дисперсия. 
# Поскольку sigma здесь это также стандартное отклонение, то большая часть значений
# будет находится в пределах +-2стандартных отклоения от среднего. 
normal_100 = stats.norm.rvs(loc=10, scale=2, size=SIZE_100, random_state=SEED)

uniform_1000 = stats.uniform.rvs(loc=5, scale=10, size=SIZE_1000, random_state=SEED)
bernoulli_1000 = stats.bernoulli.rvs(p=0.7, size=SIZE_1000, random_state=SEED)
binominal_1000 = stats.binom.rvs(n=20, p=0.6, size=SIZE_1000, random_state=SEED)
normal_1000 = stats.norm.rvs(loc=10, scale=2, size=SIZE_1000, random_state=SEED)

# Выборочное среднее
def sample_average(sampling):
	sample_average = sum([x for x in sampling]) / len(sampling)
	return sample_average

# Выборочная дисперсия
def sample_variance(sampling):
	# Выборочное среднее 
	avg = sample_average(sampling)
	sample_variance = sum([(x - avg)**2 for x in sampling]) / (len(sampling) - 1)
	return sample_variance

# Выборочное стандартное отклонение
def sample_standard_deviation(sampling):
	var = sample_variance(sampling)
	return math.sqrt(var)

samplings = [uniform_100, bernoulli_100, binominal_100, normal_100, uniform_1000, bernoulli_1000, binominal_1000, normal_1000]
pos = [i for i in range(1, 9)]
distribution_names = [
    'Uniform 100', 'Bernoulli 100', 'Binomial 100', 'Normal 100',
    'Uniform 1000', 'Bernoulli 1000', 'Binomial 1000', 'Normal 1000'
]
my_sample_averages = [sample_average(sampling) for sampling in samplings]
my_sample_variances = [sample_variance(sampling) for sampling in samplings]
my_sample_standard_deviations = [sample_standard_deviation(sampling) for sampling in samplings]

numpy_sample_averages = [np.mean(sampling) for sampling in samplings]
# ddof: N-ddof — поправка Бесселя, чтобы убрать смещение оценки. 
numpy_sample_variances = [np.var(sampling, ddof=1) for sampling in samplings]
numpy_sample_standard_deviations = [np.std(sampling, ddof=1) for sampling in samplings]

# Создание таблицы
results_df = pd.DataFrame({
    '№ выборки': pos,
    'Тип выборки': distribution_names,
    'Среднее (своё)': my_sample_averages,
    'Дисперсия (своя)': my_sample_variances,
    'Стандартное отклонение (своё)': my_sample_standard_deviations,
    'Среднее (numpy)': numpy_sample_averages,
    'Дисперсия (numpy)': numpy_sample_variances,
    'Стандартное отклонение (numpy)': numpy_sample_standard_deviations
})

# Создаю csv файл
results_df.to_csv('sampling_result.csv', index=False)


# Стиль для графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

# Создаем сетку графиков 2x4 (2 строки, 4 графика на строку)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Эмпирические и теоретические распределения')

# Параметры распределений
uniform_params = {'loc': 5, 'scale': 10}
bernoulli_params = {'p': 0.7}
binomial_params = {'n': 20, 'p': 0.6}
normal_params = {'loc': 10, 'scale': 2}

# Массивы выборок и их параметры для построения
distributions = [
    (uniform_100, 'Равномерное (100)', stats.uniform, uniform_params, 0, 0),
    (bernoulli_100, 'Бернулли (100)', stats.bernoulli, bernoulli_params, 0, 1),
    (binominal_100, 'Биномиальное (100)', stats.binom, binomial_params, 0, 2),
    (normal_100, 'Нормальное (100)', stats.norm, normal_params, 0, 3),
    (uniform_1000, 'Равномерное (1000)', stats.uniform, uniform_params, 1, 0),
    (bernoulli_1000, 'Бернулли (1000)', stats.bernoulli, bernoulli_params, 1, 1),
    (binominal_1000, 'Биномиальное (1000)', stats.binom, binomial_params, 1, 2),
    (normal_1000, 'Нормальное (1000)', stats.norm, normal_params, 1, 3)
]

for data, title, dist, params, row, col in distributions:
    ax = axes[row, col]
    
    # Построение гистограммы (плотность относительных частот)
    if dist == stats.bernoulli:
        # Для Бернулли особая обработка из-за дискретности (0,1).
				# hist_values — содержит значения высот каждого столбца гистограмма.
				# bins — массив границ интервалов столбцов гистограммы. 
        hist_values, bins, _ = ax.hist(data, bins=[-0.5, 0.5, 1.5], density=True, 
                                      alpha=0.6, label='Гистограмма', color='skyblue', edgecolor='black')
        x = np.array([0, 1])
        y = np.array([1 - params['p'], params['p']])
        ax.stem(x, y, linefmt='r-', markerfmt='ro', basefmt='r-', 
                label='Теоретическая плотность')
    elif dist == stats.binom:
        # Для биномиального распределения (дискретное)
        hist_values, bins, _ = ax.hist(data, bins=np.arange(-0.5, 21.5, 1), 
                                      density=True, alpha=0.6, label='Гистограмма', color='skyblue', edgecolor='black')
        x = np.arange(0, 21)
        y = dist.pmf(x, **params)
        ax.stem(x, y, linefmt='r-', markerfmt='ro', basefmt='r-', 
                label='Теоретическая плотность')
    else:
        # Для непрерывных распределений (равномерное, нормальное)
        hist_values, bins, _ = ax.hist(data, bins=30, density=True, 
                                      alpha=0.6, label='Гистограмма', color='skyblue', edgecolor='black')
        
        # Построение теоретической плотности вероятности
        x = np.linspace(min(data), max(data), 1000)
        y = dist.pdf(x, **params)
        ax.plot(x, y, 'r-', linewidth=2, label='Теоретическая плотность')
    
    ax.set_title(title)
    ax.set_xlabel('Значение')
    ax.set_ylabel('Плотность вероятности')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.7)
    # ax.grid(True, alpha=0.3)

# Функция устраняет перекрытие подграфиков, предотвращает обрезание подписей и заголовков, оптимизирует использование пространства.
plt.tight_layout()
# Чтобы общий заголовок не перекрывал графики
plt.subplots_adjust(top=0.9)  
# Сохраняем гистограмму в файл
plt.savefig('distributions.png', dpi=300)  
plt.show() 