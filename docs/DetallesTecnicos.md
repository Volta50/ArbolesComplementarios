# Profundización técnica del algoritmo (Greedy + Backtracking) — explicación detallada

Aquí tienes una explicación técnica y profunda de cómo funciona el algoritmo híbrido, por qué cada paso existe, garantías formales (qué cubre y qué no), análisis de complejidad, casos patológicos, y mejoras/optimizaciones prácticas. Está pensado para que lo copies en tu documentación técnica o para implementarlo con plena comprensión.

---

## 1 — Problema formalizado

Dado un árbol binario (T) con valores únicos en nodos y operación **chop(parent, side)** que elimina exactamente el hijo `side ∈ {L,R}` de `parent` (sin mover nodos), una **configuración** es un conjunto de chops aplicado a (T). Para una configuración, definimos `Leaves(T, config)` como la secuencia de hojas resultante de izquierda a derecha.

Dado dos árboles (A) y (B), queremos encontrar una secuencia (S = [s_0, s_1, \dots, s_{k-1}]) tal que:

* existe `config_A` con `Leaves(A, config_A) = S`, y
* existe `config_B` con `Leaves(B, config_B) = S`.

Buscamos la (S) de mayor longitud (k) (primero prioridad por (k), luego cualquier criterio secundario).

---

## 2 — Estructuras de datos principales

* **Representación Node**: estructura de árbol clásica con `val, left, right`. Se usa para simulaciones exactas y backtracking.
* **Representación level-order array**: lista donde el hijo izquierdo de índice `i` está en `2*i+1`, derecho en `2*i+2`. Se usa para el *greedy* por ser simple y permitir búsquedas por índice/valor rápidas.
* **Deletions set**: conjunto de strings `"{parent_val}.L"` / `"{parent_val}.R"` que indican aristas eliminadas.
* **Mapa validado (`small_map`)**: `map[k][leaf_tuple] -> set(deletion_sets)` para el árbol más pequeño. Solo contiene **pares validados**, es decir, `(tuple, deletions)` tales que aplicar `deletions` al árbol produce exactamente `tuple`. Se construye con `gen_pairs` + verificación.

---

## 3 — Pipeline general (resumen breve)

1. Construir árboles `A` y `B` desde arrays level-order.
2. Calcular hojas originales; elegir **small** (menos hojas) y **large**.
3. **Enumerar** (símbolicamente) todas las secuencias alcanzables del `small` (`gen_pairs`) y **validarlas** (simulación).
4. Ordenar candidatos por `k` descendente.
5. Para cada candidato `S`:

   * quick prechecks (existencia de valores en `large`, orden in-order creciente).
   * intentar **greedy** sobre arreglo de `large` (rápido). Si devuelve deletions, verificar en el árbol Node.
   * si greedy falla, fallback a **backtracking** (completo) con timeout.
6. Devolver la primera coincidencia válida.

---

## 4 — Generación simbólica y validación (por qué es necesaria)

`gen_pairs(node)` recorre el árbol bottom-up y para cada subárbol devuelve un conjunto de pares `(leaf_tuple, deletions)` *locales* posibles. Las combinaciones locales se componen para la raíz. Esto evita enumerar todas las (3^{n}) combinaciones globales: se factoriza por subárboles y se reutiliza.

**Crítico**: las parejas producidas por composición local pueden corresponder a conjuntos de deletions que no producen exactamente `leaf_tuple` cuando se aplican al árbol entero, por causas de interacciones en ancestros. Por eso hay que **validar** cada `(tuple, deletions)` aplicándolo al árbol entero y mantener solo los consistentes. Eso garantiza que `small_map` solo contiene candidatos reales y evita falsos positivos.

Pseudocódigo (esquemático):

```text
gen_pairs(node):
  if node is None: return { ( (), {} ) }
  if leaf: return { ( (node.val,), {} ) }
  L = gen_pairs(node.left) if left else { ((), {}) }
  R = gen_pairs(node.right) if right else { ((), {}) }
  result = {}
  for (lt, dl) in L:
    for (rt, dr) in R:
      # keep both
      result.add( (lt + rt, dl ∪ dr) )
  if left:
    for (rt, dr) in R:
      result.add( (rt, dr ∪ {node.val.L}) )
  if right:
    for (lt, dl) in L:
      result.add( (lt, dl ∪ {node.val.R}) )
  # both removed => parent as leaf
  result.add( ((node.val,), {maybe node.val.L, node.val.R}) )
  return result
```

Luego `build_validated_map(root)` recorre `raw_pairs` y sólo guarda aquellos donde `simulate(root, deletions) == tuple`.

---

## 5 — Prechecks (poda barata)

Antes de lanzar una búsqueda costosa sobre `large` para un candidato `S`, se usan filtros lineales:

1. **Existencia**: cada valor `s ∈ S` debe aparecer en `large_arr` (O(1) con mapa `val2idx`).
2. **Orden relativo in-order**: los nodos de `S` en el árbol `large` deben respetar el orden in-order del árbol (no se puede reordenar con chops; sólo se eliminan). Calculamos `pos[v]` por un recorrido inorder sobre `large_root`. Si `pos[s_i] >= pos[s_{i+1}]` para algún `i`, `S` es imposible. Este filtro descarta muchas candidaturas antes de backtracking.

---

## 6 — Greedy (fast-path) — detalle y limite

### Idea

Tomando `S = [s_0, s_1, ...]` y `large_arr`, procesar `s_i` en orden izquierdo→derecho. Para cada `s_i`:

* localizar índice `idx` con `arr[idx] == s_i`.
* Asegurarse que `s_i` es **alcanzable** (ningún ancestro ha sido desconectado por deletions previas).
* Hacer `s_i` hoja cortando sus hijos (si tiene).
* Ahora, comprobar la lista actual de hojas `curr_leaves` (simulada bajo deletions actuales). Si entre las hojas ya fijadas (prefix length = `i`) y `s_i` hay alguna hoja no fijada antes de `s_i`, eliminarla iterativamente cortando el enlace del padre que causa esa hoja (siempre que al cortar no se dañe una hoja ya fijada).
* Si en cualquier momento no es posible (p. ej. la hoja a eliminar es raíz o pertenece al subárbol de una hoja ya fijada), fallar y devolver `None`.

### Correctitud y limitaciones

* **Correcto como heurística**: si devuelve deletions y la verificación (simulación en Node) confirma `Leaves(large, deletions) == S`, entonces es una solución válida.
* **No completa**: greedy puede fallar aunque exista una solución global. Ejemplo tipológico: para eliminar varias hojas indeseadas conviene cortar una arista en un ancestro común; greedy intenta eliminarlas una por una y puede encontrarse con que hacerlo rompe hojas ya fijadas. El backtracking puede elegir una estrategia diferente (p. ej. cortar más arriba antes de fijar una hoja) que hace posible `S`.

### Pseudocódigo esquemático (ya ampliado en el script):

Ver `greedy_match_using_array(arr, target)`.

---

## 7 — Backtracking completo — cómo y por qué funciona

### Idea

El matcher recursivo intenta, en cada nodo `n` del árbol `large`, todas las combinaciones locales coherentes con un *slice* `target[i:j]` de la secuencia objetivo. Usa memoización para evitar reexplorar el mismo par `(node, start_index)`.

Casos en un nodo `n` al intentar consumir `target[i:]`:

1. Si `n` es hoja: se puede consumir `target[i]` sólo si `target[i] == n.val`.
2. Si ambos hijos ausentes: similar.
3. Si se **eliminan ambos** hijos (convertir `n` en hoja): consume `target[i]` si `target[i] == n.val`.
4. Si se **mantiene** ambos hijos: hay que elegir un split `i → m → j` tal que `left` consuma `target[i:m]` y `right` consuma `target[m:j]`.
5. Si se **elimina** uno de los hijos, el otro debe consumir todo el prefijo correspondiente.

Esto es una búsqueda sobre estructura que explora las particiones válidas del target y las combinaciones keep/remove en cada nodo. Está garantizada la completitud: si existe un conjunto de chops que produce exactly `S`, el backtracking lo encontrará (asumiendo tiempo suficiente).

### Memoización

Clave: `memo[(node_id, start_index)] = list_of_(end_index, deletions_sets)` para reutilizar subresultados. Sin memoización la explosión combinatoria vuelve inabordable.

### Timeout y ejecución en proceso separado

Como backtracking puede ser costoso en ciertos árboles, lo ejecutamos en proceso separado con `ProcessPoolExecutor` y un `timeout`. Si se excede, se aborta esa candidata y se sigue (para evitar bloqueo).

---

## 8 — Correctitud y pruebas de completitud

* **Generación/validación (`small_map`)**: garantiza que todos los candidatos `S` provienen de configuraciones reales de `small` (no falsos positivos).
* **Greedy**: es sólo heurística. Si devuelve solución y la verificación pasa, es correcta.
* **Backtracking**: completa — si existe `config_large` con `Leaves(large, config_large) == S`, el matcher lo encontrará (teorema por inducción en la estructura del árbol y particiones del target).
* **Algoritmo híbrido** encuentra la **primera** solución (mayor `k`) porque itera `k` en orden descendente y acepta la primera coincidencia verificada.

---

## 9 — Complejidad (estimaciones y comportamiento práctico)

* Sea (n) número de nodos del `small` y (m) del `large`.
* **Generación simbólica `gen_pairs`**:

  * En el peor caso produce un número de pares que crece exponencialmente con el número de nodos internos del `small` (cada nodo puede producir hasta ~4 combinaciones).
  * En práctica, deduplicación (tuplas iguales) y la validación reducen mucho la cantidad.
* **Greedy**:

  * Por candidato, coste ≈ O(m) para calcular `curr_leaves` repetidamente y localizar padres/índices; si se implementa con memos y arreglos auxiliares puede estar cerca de O(m)–O(m·k).
* **Backtracking**:

  * Peor caso exponencial: número de subproblemas bounded por O(m × k) estados (node × start_index), pero cada estado puede combinar en muchas formas — complejidad alta. Memoización reduce reexploración.
* **Estrategia híbrida**:

  * Coste real = suma sobre candidatos de (coste_prechecks + coste_greedy) + coste_backtracking × (#candidatos donde greedy falla).
  * Si greedy acierta en muchos candidatos, ahorro grande.

---

## 10 — Ejemplo que ilustra por qué greedy puede fallar (construye mental)

Árbol pequeño produce S = [a, b]. En `large`, para conseguir `a` como primera hoja es necesario **cortar un ancestro X** que eliminaría un conjunto de hojas no deseadas, pero cortar X también eliminaría un nodo que greedy desea fijar más tarde. Greedy intenta eliminar hojas no deseadas una por una (cortando hijos directos), pero ninguna combinación local permite quitar todas sin afectar futuros fijados. Backtracking, en cambio, puede decidir **antes** fijar otra elección en una parte diferente del árbol o realizar un corte en un ancestro que globalmente habilita la configuración.

(Específico: un dibujo sería ideal — en la práctica para comprender, construye un `large` donde dos hojas no deseadas comparten ancestro cercano con una hoja deseada; greedy se queda bloqueado.)

---
