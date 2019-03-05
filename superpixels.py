import numpy as np


def extract_superpixels(img, eps):
    """
    Функция выполняет разбиение изображения на суперпиксели
    Входное изображение содержит произвольное число каналов
    eps - параметр алгоритма
    """
    #print('Процесс извлечения суперпикселей начат...')
    #print('Значение параметра eps: ' + str(eps))
    #print('Размер изображения: ' + str(img.shape) + '\nВсего пикселей в изображении: ' + str(img.size))
    ch_count = 1  # количество каналов в изображении
    label = np.full((img.shape[0], img.shape[1]), -1, dtype=int)  # разметка на суперпиксели
    segments = (np.zeros((1, 1, (1 + ch_count * 2)), dtype=int))  # таблица характеристик суперпикселей
    segment_num = 0  # нумерация суперпикселей начинается с нуля
    vector_segment = (np.zeros((1, 1, (1 + ch_count * 2)), dtype=int))

    for i in range(img.shape[0]):
        #print(i)
        for j in range(img.shape[1]):
            im_vector = img[i, j]

            # обработка не левых и не верхних элементов
            if (i > 0) and (j > 0):
                left_label = label[i, j - 1]
                upper_label = label[i - 1, j]
                new_label_left = segments[left_label, 0, 0]
                new_label_upper = segments[upper_label, 0, 0]
                min_segment_new_v_left = np.minimum(segments[new_label_left, 0, 1:(ch_count + 1)], im_vector)
                max_segment_new_v_left = np.maximum(segments[new_label_left, 0, (ch_count + 1):], im_vector)
                min_segment_new_v_upper = np.minimum(segments[new_label_upper, 0, 1:(ch_count + 1)], im_vector)
                max_segment_new_v_upper = np.maximum(segments[new_label_upper, 0, (ch_count + 1):], im_vector)
                condition_upper = np.all((max_segment_new_v_upper - min_segment_new_v_upper) <= 2*eps)
                condition_left = np.all((max_segment_new_v_left - min_segment_new_v_left) <= 2*eps)

                # номера сегментов соседей совпадают
                if new_label_left == new_label_upper:
                    # проверка условия для верхнего ( == левого) сегмента
                    if np.all((max_segment_new_v_upper - min_segment_new_v_upper) <= 2*eps):
                        label[i, j] = new_label_upper
                        # меняем информацию о сегменте
                        segments[new_label_upper, 0, (ch_count + 1):] = max_segment_new_v_upper
                        segments[new_label_upper, 0, 1:(ch_count + 1)] = min_segment_new_v_upper

                    else:
                        segment_num += 1
                        label[i, j] = segment_num
                        vector_segment[0, 0, :] = np.hstack((segment_num, im_vector, im_vector))
                        segments = np.vstack((segments, vector_segment))

                # номера сегментов соседей не совпадают
                elif new_label_left != new_label_upper:
                    # условие выполняется только для верхней области
                    if condition_upper and not(condition_left):
                        label[i, j] = new_label_upper
                        # меняем информацию о сегменте
                        segments[new_label_upper, 0, 1:(ch_count + 1)] = min_segment_new_v_upper
                        segments[new_label_upper, 0, (ch_count + 1):] = max_segment_new_v_upper
                    # условие выполняется только для левой области
                    elif condition_left and not(condition_upper):
                        label[i, j] = new_label_left
                        # меняем информацию о сегменте
                        segments[new_label_left, 0, 1:(ch_count + 1)] = min_segment_new_v_left
                        segments[new_label_left, 0, (ch_count + 1):] = max_segment_new_v_left
                    # условие выполняется для обеих областей
                    elif condition_upper and condition_left:
                        # проверка на то, можно ли слить области
                        m = (np.maximum(max_segment_new_v_left, max_segment_new_v_upper) - np.minimum(min_segment_new_v_left, min_segment_new_v_upper))
                        if np.all(m <= 2*eps):
                            # будем объединять в область с минимальным label
                            new_min_label = min(new_label_left, new_label_upper)
                            new_max_label = max(new_label_left, new_label_upper)
                            # добавили в эту область рассматриваемый пиксель
                            label[i, j] = new_min_label
                            # выполнили слияние областей - заменили все номера
                            segments[(segments[:, 0, 0] == new_max_label), 0, 0] = new_min_label
                            # поменяли характеристики сегмента
                            segments[new_min_label, 0, 1:(ch_count + 1)] = np.minimum(min_segment_new_v_upper, min_segment_new_v_left)
                            segments[new_min_label, 0, (ch_count + 1):] = np.maximum(max_segment_new_v_upper, max_segment_new_v_left)
                        # если слилась только одна вершина, то смотрим на сколько ближе каждый из векторов
                        else:
                            if np.all(sum(abs((max_segment_new_v_upper + min_segment_new_v_upper)/2 - im_vector)) <= sum(abs((max_segment_new_v_left + min_segment_new_v_left)/2 - im_vector))):
                                label[i, j] = new_label_upper
                                segments[new_label_upper, 0, 1:(ch_count + 1)] = min_segment_new_v_upper
                                segments[new_label_upper, 0, (ch_count + 1):] = max_segment_new_v_upper
                            else:
                                label[i, j] = new_label_left
                                segments[new_label_left, 0, 1:(ch_count + 1)] = min_segment_new_v_left
                                segments[new_label_left, 0, (ch_count + 1):] = max_segment_new_v_left
                    # условие не выполняется ни для одной из областей, назначается новая область
                    else:
                        segment_num += 1
                        label[i, j] = segment_num
                        vector_segment[0, 0, :] = np.hstack((segment_num, im_vector, im_vector))
                        segments = np.vstack((segments, vector_segment))

            # обработка левого столбца, кроме верхнего левого элемента
            elif (i > 0) and (j == 0):
                upper_label = label[i - 1, j]
                # если было объединение сегментов, то сегмент мог получить новый номер
                new_label_upper = segments[upper_label, 0, 0]
                min_segment_new_v = np.minimum(segments[new_label_upper, 0, 1:(ch_count + 1)], im_vector)
                max_segment_new_v = np.maximum(segments[new_label_upper, 0, (ch_count + 1):], im_vector)
                if np.all((max_segment_new_v - min_segment_new_v) <= 2*eps):
                    label[i, j] = new_label_upper
                    # меняем информацию о сегменте
                    segments[new_label_upper, 0, 1:(ch_count + 1)] = min_segment_new_v
                    segments[new_label_upper, 0, (ch_count + 1):] = max_segment_new_v
                # для какого-либо из каналов условие не выполняется -> создаем границу (создаем новый сегмент)
                else:
                    segment_num += 1
                    label[i, j] = segment_num
                    vector_segment[0, 0, :] = np.hstack((segment_num, im_vector, im_vector))
                    segments = np.vstack((segments, vector_segment))

            # обработка верхней строки, кроме верхнего левого элемента
            elif (i == 0) and (j > 0):
                left_label = label[i, j - 1]
                # если ранее было объединение сегментов, то сегмент мог получить новый номер
                new_label_left = segments[left_label, 0, 0]
                min_segment_new_v = np.minimum(segments[new_label_left, 0, 1:(ch_count + 1)], im_vector)
                max_segment_new_v = np.maximum(segments[new_label_left, 0, (ch_count + 1):], im_vector)
                if np.all((max_segment_new_v - min_segment_new_v) <= 2*eps):
                    label[i, j] = new_label_left
                    # меняем информацию о сегменте
                    segments[new_label_left, 0, 1:(ch_count + 1)] = min_segment_new_v
                    segments[new_label_left, 0, (ch_count + 1):] = max_segment_new_v
                # если хотя бы для одного из каналов условие не выполняется -> создаем границу (создаем новый сегмент)
                else:
                    segment_num += 1
                    label[i, j] = segment_num
                    vector_segment[0, 0, :] = np.hstack((segment_num, im_vector, im_vector))
                    segments = np.vstack((segments, vector_segment))

            # обработка верхнего левого элемента
            elif (i == 0) and (j == 0):
                label[i, j] = segment_num
                segments[0, 0, :] = np.hstack((segment_num, im_vector, im_vector))

            # для проверки
            else:
                label[i, j] = -100

    return label, segments


def update_label(segments, label):
    """
    Функция выполняет замену значений в таблице label
    с индексов segments на новые значения, хранящиеся в нулевом канале
    """
    #print('Начался процесс обновления меток...')
    for i in range(len(segments)):
        if (i != segments[i, 0, 0]):
            label[label == i] = segments[i, 0, 0]
    #print('Процесс обновления меток окончен.')
    return label


def delete_rows(segments):
    """
    Функция удаляет строки с одинаковыми значениями номера сегмента
    """
    ch_count = int((segments.shape[2]-1)/2)
    #print('Началось удаление неактуальных строк из таблицы характеристик...')
    a = segments[0]
    for i in range(1, len(segments)):
        if (i == segments[i, 0, 0]):
            a = np.vstack((a, segments[i]))
    #print('Строки удалены.')
    # массив b нужен для перестановки столбцов
    b = np.zeros(a.shape, dtype=int)
    b[:, 0] = a[:, 0]
    for k in range(1, ch_count + 1):
        b[:, 2*k - 1] = a[:, k]
        b[:, 2*k] = a[:, k + ch_count]
    return b


def update_label1(segments, label):
    """
    Функция выполняет замену значений в таблице label
    на индексы новые индексы segments
    """
    #print('Начался процесс обновления меток...')
    for i in range(len(segments)):
        label[label == segments[i, 0]] = i
    #print('Процесс обновления меток окончен.')
    #print('Размер таблицы харакетристик: ' + str(segments.shape) + '\n')
    return label

