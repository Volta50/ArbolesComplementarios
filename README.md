# ÁrbolesComplementarios
## Integrantes
- Santiago Quevedo (Volta50)
- Sivano Vargas (sivargasg)
- Ariel Cárdenas (ArielDcg)
---

## Contenido

* `match_both_cli_hybrid.py` — Implementación híbrida (greedy fast-path + backtracking con timeout y prechecks).
* `match_both_benchmark.py` — Script de benchmarking (compara híbrido vs backtracking-only).
* Ejemplo de entrada: `arrays.txt` (dos líneas, una por cada árbol en forma de array level-order).

---
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
py match_both_cli_hybrid.py --file arrays.txt
```
Especificar en terminal
```bash
py match_both_cli_hybrid.py --a "[1,2,3,None]" --b "1,2,3,None"
```
Guadar plots en vez de mostrarlos
```bash
py match_both_cli_hybrid.py --file arrays.txt --save-dir out_plots
```

**NB:** correr `py --version` para verificar la versión de Python en Windows y el prefijo `py`


Opciones principales:

* `--file, -f PATH` — archivo con dos arrays (A luego B).
* `--a STR` / `--b STR` — arrays como listas (ej. `"[1,2,3,None]"`) o coma-separados.
* `--save-dir DIR` — guarda las visualizaciones (PNG) en `DIR`.
* `--no-greedy` — desactiva el greedy; fuerza solo backtracking.
* `--bt-timeout N` — timeout en segundos para backtracking por candidato (default `3.0`).

### Visualización
Se mostrarán los dos árboles originales, oprima **Enter** para que se muestre los árboles complementarios resultantes.
**Ejemplo:**
**Árboles originales**  

<img width="600" height="300" alt="orig_A" src="https://github.com/user-attachments/assets/682e493c-74dd-4639-b68b-4b8993362cec" />

<img width="600" height="300" alt="orig_B" src="https://github.com/user-attachments/assets/c0e76680-316d-4dad-8aa8-d85d465a7115" />

**Árboles finales complementarios**  

<img width="600" height="300" alt="final_A" src="https://github.com/user-attachments/assets/bdd62d70-67c6-4c91-b917-96b17e7095b5" />

<img width="600" height="300" alt="final_B" src="https://github.com/user-attachments/assets/76d71a3f-1927-41b1-aa2b-fd8a78303922" />

---
---
## Desarrollo del programa
Esta implementación busca una secuencia común de hojas (izquierda→derecha) alcanzable por chops (eliminación de aristas hijo) en dos árboles binarios.
Estrategia híbrida:

1. Enumerar todas las secuencias de hojas alcanzables del árbol más pequeño (mapa validado).

2. Para cada secuencia (mayor k → menor k):

   - intentar primero un greedy fast-path (operando sobre el arreglo por niveles del árbol grande),
   - si falla, ejecutar backtracking exhaustivo con timeout (proceso separado),
   - aceptar la primera coincidencia válida y terminar.

Esto combina velocidad (greedy) y completitud (backtracking).




### Representación inicial

Cada árbol se representa como un arreglo en orden por niveles (level-order array), donde los valores `None` indican ausencia de nodo.

Ejemplo:

```python
# Árbol A
[1, 2, 7, 3, 6, None, 8, 4, 5, None, None, None, None]

# Árbol B
[100, 50, 150, 25, 75, 125, 175, 10, None, 65, None, None, None, None, 85]
```

### Pasos del algoritmo
Para una explicación más detallada dirigerse a [Detalles Técnicos](docs\DetallesTecnicos.md)
#### 1. Determinar el árbol base

1. Contar el número de hojas de cada árbol.
2. Escoger el árbol con menos hojas como base.
   Esto reduce el espacio de búsqueda, ya que sólo es necesario intentar reproducir las hojas del árbol más pequeño en el árbol más grande.

#### 2. Generar posibles secuencias de hojas

1. Para el árbol base, generar todas las secuencias posibles de hojas que pueden obtenerse aplicando operaciones de *chop*.
2. Estas secuencias se ordenan de mayor a menor longitud, de modo que el algoritmo intente primero las coincidencias con más hojas.

#### 3. Intentar coincidencia con el árbol grande (método *greedy*)

Para cada secuencia candidata del árbol base:

1. Tomar la secuencia de hojas objetivo (por ejemplo `[10, 85]`).
2. Recorrer el árbol grande (su arreglo nivel por nivel).
3. Buscar el primer elemento de la secuencia dentro del árbol.
4. Aplicar las operaciones de *chop* necesarias para que dicho nodo se convierta en hoja, sin alterar el orden previo de las hojas que ya fueron fijadas.
5. Continuar con el siguiente elemento de la secuencia:

   * Si es posible hacerlo sin perder el orden de las hojas anteriores, continuar.
   * Si no es posible, descartar esta secuencia y probar con la siguiente.
6. Si se logra construir toda la secuencia de hojas deseada, se registra el conjunto de *chops* aplicados.
#### 4. Fallback backtracking:

   * Si greedy falla, ejecutar matcher exhaustivo (recursivo + memoización) en proceso separado con timeout.
   * Cachear resultados por `(arr_large, target)` para evitar recalcular.
#### 5. Validar coincidencia

1. Si ambos árboles pueden producir la misma secuencia de hojas con algún conjunto de *chops*, esa secuencia se considera una coincidencia válida.
2. El algoritmo termina en cuanto se encuentra la primera coincidencia (la de mayor número de hojas).
3. Si no se encuentra coincidencia, se repite el proceso con secuencias más cortas hasta que se encuentre una coincidencia válida o no haya más secuencias que probar.

#### 6. Visualización (opcional)

El algoritmo puede mostrar:

* El árbol original y el árbol final después de aplicar los *chops*.
* La secuencia de hojas resultante.
* El conjunto de *chops realizados*, representado por las conexiones eliminadas (`nodo.hijo`).







---


## Consideraciones de rendimiento

* **Greedy**: muy rápido en la práctica (O(n)–O(n²) por candidato), pero incompleto (falsos negativos posibles).
* **Backtracking**: completo (garantiza encontrar solución si existe) pero exponencial en peor caso.
* **Híbrido**: combinación recomendada; evita la mayoría de backtracks innecesarios.
* Optimización añadidas:

  * prechecks (existencia + inorder) para podar candidatos,
  * timeout por candidato (`--bt-timeout`),
  * cache de backtracking,
  * ordenar candidatos por `k` descendente,
  * posibilidad de desactivar greedy (`--no-greedy`) para verificación exhaustiva.

---

## Benchmark
`match_both_benchmark.py` compara el tiempo que toma con el enfoque híbrido y solamente utilizando backtracking, para el ejemplo utilizado anteriormente se obtuvo este resultado:
 ```bash
 Running benchmark...
===== BENCHMARK SUMMARY =====
Candidates tested (limit): 10

HYBRID (greedy first, fallback to backtracking):
  total time: 0.000s
  candidates tried: 3
  greedy successes: 1
  backtracking calls: 0
  backtracking successes: 0
  avg time/candidate: 0.0000s
  found target (method): (2, -1) greedy
  deletions small: {'4.R'}
  deletions large: {'7.R', '5.L', '7.L', '3.L', '-2.R'}

BACKTRACKING-ONLY:
  total time: 0.529s
  candidates tried: 3
  backtracking calls: 1
  backtracking successes: 1
  avg time/candidate: 0.1764s
  found target: (2, -1)
  deletions small: {'4.R'}
  deletions large: {'5.L'}
=============================
 ```

Esto demuestra la eficiencia del método híbrido.


Además puede ser utilizado para entender fallos del greedy en casos concretos, usar `--no-greedy` y revisar la salida del backtracking.

---



## Estructuras de datos utilizadas

El algoritmo híbrido combina dos estrategias complementarias (Greedy y Backtracking), y para ambas se utilizan estructuras de datos cuidadosamente elegidas que facilitan la representación de los árboles, la búsqueda de hojas, y la simulación de los "chops" (cortes de ramas).

A continuación se explican las estructuras clave y el motivo de su uso:

---

### 1. Representación de árboles: **Listas en orden por niveles (Level-Order Arrays)**

**Estructura:**
Cada árbol binario se almacena como una **lista de Python**, donde cada elemento representa un nodo y su posición en la lista define su relación padre-hijo.

Por convención:

* Hijo izquierdo: índice `2*i + 1`
* Hijo derecho: índice `2*i + 2`

**Ejemplo:**

```python
tree = [50, 30, 75, 20, 40, 60, 85, None, 25, 35, None, None, 10, None, None]
```

**Ventajas:**

* Muy eficiente para **recorrer niveles** y localizar hijos/padres sin estructuras adicionales.
* Permite una **traducción directa de los árboles desde archivos o JSON**.
* Ideal para **simular cortes** (eliminar ramas completas reemplazando nodos por `None`).

**Por qué se eligió:**
Porque permite trabajar con árboles de diferentes tamaños sin estructuras dinámicas ni punteros, manteniendo las operaciones de indexado y copia muy ligeras.

---

### 2. Conjunto de cortes (“chops”): **`set[str]`**

Cada operación de poda o eliminación se representa como una cadena `"nodo.dirección"`, por ejemplo:

```python
{"50.L", "150.L"}
```

**Uso:**

* El conjunto se actualiza a medida que se aplican cortes.
* Se utiliza tanto en greedy como en backtracking.
* El uso de un `set` evita duplicados y permite comprobaciones rápidas (`O(1)`) de si un chop ya fue aplicado.

**Ventajas:**

* Rápido acceso y comparación entre diferentes soluciones parciales.
* Ideal para guardar el “estado actual” de poda sin estructuras complejas.

---

### 3. Estado del árbol en ejecución: **Listas temporales (`list`)**

Durante la ejecución del algoritmo, especialmente en el backtracking, se mantienen copias del árbol con los cortes aplicados.
Estas copias son listas derivadas del árbol original (con nodos reemplazados por `None`).

**Ejemplo:**

```python
temp_tree = apply_chops(original_tree, {"50.R", "150.L"})
```

**Ventajas:**

* Permite simular cambios de estructura sin afectar el árbol original.
* Facilita la exploración recursiva en el backtracking (copiar y revertir estados).

---

### 4. Cache de hojas y subproblemas: **`dict` (memoización)**

En el backtracking se emplea un **diccionario de memoización**:

```python
cache: dict[tuple, list] = {}
```

Las claves son representaciones inmutables del estado (por ejemplo, una tupla del árbol actual o de los chops aplicados), y el valor es el resultado ya calculado.

**Uso:**

* Si una configuración de cortes ya fue analizada, se reutiliza el resultado.
* Reduce exponencialmente el número de combinaciones exploradas.

**Ventajas:**

* Evita recomputar subárboles idénticos.
* Disminuye drásticamente el tiempo de búsqueda en árboles medianos o grandes.

---

### 5. Colas y pilas (implícitas) para búsqueda

* **Greedy:** usa un bucle secuencial (implícitamente una cola) para recorrer hojas y eliminar nodos.
* **Backtracking:** utiliza la **pila de llamadas recursivas** de Python como estructura de exploración en profundidad.

**Motivo:**

* No es necesario implementar estructuras adicionales: la recursión natural de Python y los bucles controlan el flujo.
* Esto mantiene el código más legible y limpio.

---

### 6. Almacenamiento de resultados y trazas: **Listas y objetos de resultados**

Cada ejecución (greedy, backtracking o híbrida) produce un objeto o diccionario con:

```python
{
  "matched": True,
  "target_leaves": [...],
  "deletions": {...},
  "runtime": 0.042,
  "method": "greedy"
}
```

**Ventajas:**

* Estandariza la salida y facilita comparar estrategias (por ejemplo, medir tiempo o número de cortes).
* Compatible con generación de reportes (`.csv`, `.json`, o visualizaciones).

---

## En resumen

| Componente                     | Estructura usada           | Motivo principal                                    |
| ------------------------------ | -------------------------- | --------------------------------------------------- |
| Árbol binario                  | `list` (level-order)       | Acceso rápido a hijos y simulación simple de cortes |
| Cortes ("chops")               | `set[str]`                 | Evitar duplicados y permitir búsqueda O(1)          |
| Árbol modificado temporalmente | `list`                     | Simulación segura de podas sin afectar el original  |
| Cache / memoización            | `dict`                     | Reutilización de resultados parciales               |
| Exploración en profundidad     | Recursión (pila implícita) | Simplicidad y claridad de implementación            |
| Resultados y métricas          | `dict` o `dataclass`       | Uniformidad y compatibilidad con el CLI y reportes  |

---

## Idea central

El diseño no busca estructuras exóticas, sino una combinación **ligera y eficiente**:

* Los árboles se manipulan como listas (`O(1)` por acceso),
* Los chops se gestionan con conjuntos,
* Las búsquedas combinatorias usan diccionarios de cache,
* Y el sistema híbrido puede alternar entre *velocidad (greedy)* y *completitud (backtracking)* sin reestructurar los datos.




