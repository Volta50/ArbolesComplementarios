# ÁrbolesComplementarios


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
**Ejemplo:**
**Árboles originales**  

<img width="600" height="300" alt="orig_A" src="https://github.com/user-attachments/assets/682e493c-74dd-4639-b68b-4b8993362cec" />

<img width="600" height="300" alt="orig_B" src="https://github.com/user-attachments/assets/c0e76680-316d-4dad-8aa8-d85d465a7115" />

**Árboles finales complementarios**  

<img width="600" height="300" alt="final_A" src="https://github.com/user-attachments/assets/bdd62d70-67c6-4c91-b917-96b17e7095b5" />

<img width="600" height="300" alt="final_B" src="https://github.com/user-attachments/assets/76d71a3f-1927-41b1-aa2b-fd8a78303922" />


## Desarrollo del programa


### Pasos del algoritmo


### Estructuras de datos empleadas
