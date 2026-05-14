cmake_minimum_required(VERSION 3.10)
project(SelfConfig-GP-for-NeuronNetwork)


set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE Debug)  # Включает отладочную информацию
#set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-O0 -g3)  # Отключаем оптимизации


add_executable(Source
    Source.cpp
    neuron_network/Tree.cpp
    SAGP/AdaptiveGeneticProgramming.cpp
    Diff_evolution/IndividualDiffEvolution.cpp
    Diff_evolution/DiffMutation.cpp
    Diff_evolution/DiffEvolution.cpp
    # Добавьте другие исходные файлы
)

# Копируем данные в папку сборки (опционально)
file(COPY test/ DESTINATION ${CMAKE_BINARY_DIR}/test)
file(COPY algorithm_results/ DESTINATION ${CMAKE_BINARY_DIR}/algorithm_results)