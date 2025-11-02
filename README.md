# ArbolesComplementarios
Solucion del problema de arboles complementarios solo podando ramas

## Manual de usuario
Para crear los árboles se especifican sus elementos en orden secuencial usando un arreglo, dado que no son necesariamente árboles completos se utiliza `None` para los elementos vacíos.  

**Ejemplo**
[6, 2, 4, None, None, -1, 5, None, None]

        6
       / \
      2   4
         / \
       -1   5

En un archivo .txt especifique dos árboles de la siguiente forma
```markdown
[6,2,4, None, None, -1, 5, None, None]
[5,7,4,3,-2,5,-1,8, None, None, 12, 10,2, None, None, None, None, None, None, 1, 2]  
```  
### Comandos
Ahora ejecute los comandos descritos a continuación  

Con archivos de texto(recomendado)
```bash
py match_both_cli.py --file arrays.txt
```
Especificar en terminal
```bash
py match_both_cli.py --a "[1,2,3,None]" --b "1,2,3,None"
```
Guadar plots en vez de mostrarlos
```bash
py match_both_cli.py --file arrays.txt --save-dir out_plots
```

**NB:** correr `py --version` para verificar la versión de Python en Windows y el prefijo `py`
### Visualización
Se mostrarán los dos árboles originales, oprima **Enter** para que se muestre los árboles complementarios resultantes.
## Desarrollo del programa


### Pasos del algoritmo


### Estructuras de datos empleadas
