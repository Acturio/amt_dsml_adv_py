# amt22_03intro2mls2
Curso intermedio de Ciencia de Datos y Machine Learning con aprendizaje supervisado II

https://acturio.github.io/amt22_03intro2mls2/

## Ambiente Python

```bash
conda env create -f environment.yml
conda activate dsml_py_adv
```

Los chunks de Python verifican este ambiente al iniciar cada archivo `.Rmd`.
El primer chunk de `index.Rmd` configura `reticulate` para usar el ambiente
conda `dsml_py_adv` durante el render de `bookdown`.
