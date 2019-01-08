Máquinas de Vector Soporte (SVM, Support Vector Machine)
========================================================

En el contexto de *Machine Learning*, las Máquinas de Vector Soporte son
modelos de aprendizaje supervisado asociados a los algoritmos que
analizan datos para su análisis de clasificación y/o regresión.

Dado un conjunto de muestras de entrenamiento, cada una de ellas
clasificada en una categoría, el algoritmo de entrenamiento de una SVM
construye un modelo que asigna una clase a cada observación. Un modelo
SVM es una representación de estas muestras en el espacio de tal manera
que las muestras de cada categoría están separadas de forma clara.

Historia
--------

El algoritmo original de las Máquinas de Vector Soporte fue escrito por
*Vladimir N. Vapnik* y *Alexey Ya. Chervonenkis* en 1963. En 1992,
*Bernhard E. Boser*, *Isabelle M. Guyon* y *Vladimir N. Vapnik*
sugirieron una metodología para crear clasificadores no lineales
aplicando el mismo concepto de hiperplanos de margen máximo.

Motivación
----------

La clasificación de muestras es una tarea común en *Machine learning* en
la que dados unos datos en los que cada observación pertenece a alguna
clase y la finalidad es decidir a qué clase asignar una nueva
observación. En el caso de las SVM, cada observación se considera un
vector de *p* dimensiones (tenemos *p* variables) y tratamos de separar
cada clase. Si lo hacemos con hiperplano de dimensiones *p-1* estaremos
aplicando un clasificador lineal. Hay muchos hiperplanos que podrían
clasificar nuestros datos. Podemos razonar y buscar aquel hiperplano que
muestra la mayor separación entre clases. Elegimos entonces aquel
hiperplano cuya distancia a los puntos más cercanos de cada lado sea
máxima. Si tal hiperplano existe se denomina *hiperplano de máximo
margen* y el clasificador lineal asociado se define como *clasificador
de máximo margen*.

Definición
----------

Podemos definir más formalmente este hiperplano. Una Máquina de Vector
Soporte construye un hiperplano o conjunto de hiperplanos
*n*-dimensionales que pueden ser usado para clasificación, regresión u
otras tareas. Geométricamente, una buena separación se alcanzará por
aquel hiperplano que tenga la mayor distancia entre las observaciones de
cada clase en las muestras de entrenamiento.

### kernel

A veces, el problema original puede ser resuelto en un espacio de
dimensión finita, pero en otras ocasiones sucede que los conjuntos a
discriminar no tienen una separación lineal en ese espacio. Para
solventar este inconveniente, el espacio de dimensión finita donde está
planteado el problema puede ser transformado a un espacio de dimensión
mayor, donde es esperable que la separación entre clases se más fácil de
calcular.

El aumentar la dimensión del espacio en el que estamos trabajando
implica un coste computacional mayor. Para que este aumento sea
razonable las transformaciones a espacios de dimensión mayor se diseñan
de tal manera que los productos escalares en estos nuevos espacios
puedan ser calculados fácilmente en términos de las variables iniciales.
Para ello se utilizan las funciones *kernel* *k*(*x*, *y*) seleccionadas
específicamente para resolver este problema. Los hiperplanos en una
mayor dimensión son definidos como aquellos conjuntos de puntos tales
que su producto escalar con un vector en ese espacio es constante. Los
vectores que definen los hiperplanos pueden ser elegidos como una
combinación lineal con parámetros *α*<sub>*i*</sub> de imágenes de los
vectores de características *x*<sub>*i*</sub>. Si elegimos el hiperplano
con estas propiedades, los puntos *x* en el espacio de características
son llevados hiperplanos que se define por la siguiente relación:

Tenemos que tener en cuenta que si *k*(*x*, *y*) se vuelve pequeño a
medida que *y* crece más lejos de *x*, cada término de la suma mide el
grado de cercanía de la prueba Punto *x* al punto de base de datos
correspondiente *x*<sub>*i*</sub>. De esta manera, la suma de los
núcleos anteriores puede usarse para medir la proximidad relativa de
cada punto de prueba a los puntos de datos originados en uno u otro de
los conjuntos a discriminar.

#### Construcción de un *kernel*

Como hemos comentado, la función *kernel* nos lleva el espacio de
características (donde están nuestros datos) a un nuevo espacio de
dimensión mayor de tal manera que sea fácil calcular el producto escalar

Sea el *kernel* definido como

que lleva un punto *x* ∈ *R*<sup>2</sup> a *z* ∈ *R*<sup>3</sup>. En la
figura observamos que este kernel separa el interior de la
circunferencia con el exterior con un hiperplano en *R*<sup>3</sup>,
siendo mucho más fácil separar las clases en las que están divididas las
observaciones.

<!-- https://www.r-bloggers.com/learning-kernels-svm/ -->
Las Máquinas de Vector Soporte funcionan agrupando los puntos de las
características según sus clases. En la figura se generan dos vectores
de características bidimensionales
*x* = {*x*<sub>1</sub>, *x*<sub>2</sub>} de tal manera que la clase
*y* =  − 1 puntos (triángulos) están bien separados de la clase *y* = 1
(círculos).

El algoritmo encuentra el mayor margen lineal posible que separa estas
dos regiones. Los separadores se apoyan sobre los puntos avanzados que
están justo en la línea frente a sus respectivas regiones. Estos puntos,
marcados como dos triángulos en negrita y un círculo en negrita en la
figura , se llaman los *vectores de apoyo* o *vectores soporte*, ya que
están apoyando las líneas de separación. De hecho, la tarea de
aprendizaje del algoritmo de Máquinas de Vector Soporte consiste en
determinar estos puntos vector de soporte y la distancia de margen que
separa las regiones. Después del entrenamiento, todos los demás puntos
de no apoyo no se usará para futuras predicciones.

En el espacio de características lineales, los vectores soporte se suman
a un vector de hipótesis general *h*,

De modo que las fronteras de clasificación están dadas por las líneas
*h**x* + *b* = 1 y *h**x* + *b* =  − 1 centradas alrededor de
*h**x* + *b* = 0.

El código en el anexo es una modificación de la implementación de la
función `ksvm()` en el paquete `kernlab` de `R`, haciendo uso de los
tutoriales de *Jean-Philippe Vert* para representar las líneas de
separación de clasificación mediante un kernel lineal.

<img src="graphics/svm/kernel_lineal-1.pdf" alt="Implementación de la función {\tt ksvm()} en el paquete {\tt kernlab} de {\tt R} para representar las líneas de separación de clasificación." width="80%" />
<p class="caption">
Implementación de la función {ksvm()} en el paquete {kernlab} de {R}
para representar las líneas de separación de clasificación.
</p>

En la figura se ilustra un ejemplo en el que las observaciones no están
separados . Los puntos de la clase *y* = 1 (círculos) se colocan en una
región interior rodeada por puntos de clase *y* =  − 1 (triángulos). En
este ejemplo no hay una sola línea recta (lineal) que pueda separar
ambas regiones. Sin embargo es posible encontrar un separador lineal
mediante la transformación de los puntos
*x* = {*x*<sub>1</sub>, *x*<sub>2</sub>} del espacio de características
a un espacio cuadrático de núcleos con puntos dados por las
correspondientes coordenadas cuadradas
{*x*<sub>1</sub><sup>2</sup>, *x*<sub>2</sub><sup>2</sup>}. El código en
`R` puede consultarse en el código en el anexo.

La técnica de transformar el espacio de características en una medida
que permite una separación lineal puede formalizarse en términos de
*kernel*. Suponiendo que *Φ*() sea una función de transformación
vectorial de coordenadas, un espacio de coordenadas cuadráticas sería
{*Φ*(*x*<sub>1</sub>), *Φ*(*x*<sub>2</sub>)} = {*x*<sub>1</sub><sup>2</sup>, *x*<sub>2</sub><sup>2</sup>}.
La búsqueda de separación de la SVM está actuando ahora en el espacio
transformado para encontrar los vectores de soporte que generan la
condición:

Para el vector de hipótesis *h*:

Dada por la suma sobre los puntos vectoriales de soporte
*x*<sub>*i*</sub>. Poniendo ambas expresiones juntas obtenemos

Con la función de kernel escalar
*K*(*x*<sub>*i*</sub>,*x*) = *Φ*(*x*<sub>*i*</sub>) ⋅ *Φ*(*x*). El
kernel se compone del producto escalar entre un vector soporte
*x*<sub>*i*</sub> y otro punto vector *x* de características en el
espacio transformado.

En la práctica, el algoritmo SVM puede expresarse completamente en
términos de kernels sin tener que especificar realmente la
transformación de espacio de entidad. Los núcleos populares son, por
ejemplo, potencias superiores del producto escalar lineal (*kernel*
polinomial). Otro ejemplo es una probabilidad pesada de distancia entre
dos puntos (*kernel* gaussiano).

La implementación de una función de núcleo cuadrático bidimensional
permite al algoritmo SVM encontrar vectores soporte y separar
correctamente las regiones. En la figura se muestra que regiones no
lineales se pueden separar linealmente después de una transformación
adecuada.

<img src="graphics/svm/kernel_cuadratico-1.pdf" alt="Implementación de una función de núcleo cuadrático bidimensional permite al algoritmo SVM encontrar vectores soporte y separar correctamente las regiones." width="95%" />
<p class="caption">
Implementación de una función de núcleo cuadrático bidimensional permite
al algoritmo SVM encontrar vectores soporte y separar correctamente las
regiones.
</p>

#### Ejemplos de funciones *kernel* más utilizadas

*kernel* lineal, el más sencillo de los posibles

*kernel* de base radial (RBF, Laplace Radial Basis Function)

*kernel* de base radial (RBF, Gaussian Radial Basis Function)

*kernel* polinomial

Aplicaciones
------------

SVMs pueden ser utilizados para resolver varios problemas del mundo
real, como por ejemplo:

-   La clasificación de las imágenes se puede realizar usando SVMs. Los
    resultados experimentales muestran que las SVM logran una precisión
    de búsqueda significativamente mayor que otros algoritmos de
    clasificación supervisada.

-   Los caracteres escritos a mano se pueden reconocer usando SVM. Estos
    algoritmos son conocidos como algoritmos
    [ocr](https://en.wikipedia.org/wiki/Optical_character_recognition).
    Una aplicación muy conocida de reconocimiento de caracteres son los
    [captchas](https://en.wikipedia.org/wiki/CAPTCHA).

SVM lineales
------------

Dado un conjunto *n* de observaciones de entrenamiento de la forma:

Donde *y*<sub>*i*</sub> toma los valores -1 o 1, indicando la clase a la
que pertenece cada punto *x⃗*<sub>*i*</sub>. Cada h *x⃗*<sub>*i*</sub>
es un vector de *p* dimensiones. Queremos encontrar el hiperplano de
margen máximo que divide el grupo de puntos *x⃗*<sub>*i*</sub> entre los
que verifican que *y*<sub>*i*</sub> = 1 de conjunto de puntos que
verifican *y*<sub>*i*</sub> =  − 1. Este hiperplano es definido como
aquel cuya distancia a los puntos más cercanos *x⃗*<sub>*i*</sub> de
cada clase es máxima.

Un hiperplano puede ser descrito como el conjunto de puntos *x⃗* que
satisfacen la siguiente condición:

Donde *w⃗* es el vector normal (no necesariamente normalizado) al
hiperplano. El parámetro
${\\displaystyle {\\tfrac {b}{\\\|{\\vec {w}}\\\|}}}$ determina el
desplazamiento de la hiperplano desde el origen a lo largo del vector
normal *w⃗*.

Problemas AND, OR y XOR
-----------------------

### Problema AND

AND es un operador lógico cuyo valor de la verdad resulta en cierto sólo
si ambas proposiciones son ciertas, y en falso de cualquier otra forma.
En la figura vemos cómo la maquina vector soporte de kernel lineal,
resuelve el problema usando tres soportes.

``` r
data.and <- data.frame(x = c(-1, -1, 1, 1), 
                   y = c(-1, 1, -1, 1),
                   class = c(-1, -1, -1, 1))

modelo <- ksvm(class ~ ., 
               data = data.and, 
               type = "C-svc", 
               kernel = "vanilladot")
```

    ##  Setting default kernel parameters

``` r
# table(predict(modelo), data.and$class)
plot(modelo, 
     data = data.and, 
     xlim = c(-1.1, 1.1), 
     ylim = c(-1.1, 1.1))
```

<img src="graphics/svm/and-1.pdf" alt="Implementación de una máquina de vector soporte para resolver el problema AND." width="65%" />
<p class="caption">
Implementación de una máquina de vector soporte para resolver el
problema AND.
</p>

### Problema OR

OR es un operador lógico que implementa la disyunción lógica y se
comporta de acuerdo a la tabla .

``` r
data.or <- data.frame(x = c(-1, -1, 1, 1), 
                       y = c(-1, 1, -1, 1),
                       class = c(-1, 1, 1, 1))

modelo <- ksvm(class ~ ., 
               data = data.or, 
               type = "C-svc", 
               kernel = "vanilladot")
```

    ##  Setting default kernel parameters

``` r
# table(predict(modelo), data.or$class)
plot(modelo, 
     data = data.or, 
     xlim = c(-1.1, 1.1), 
     ylim = c(-1.1, 1.1))
```

<img src="graphics/svm/or-1.pdf" alt="Implementación de una máquina de vector soporte para resolver el problema OR." width="65%" />
<p class="caption">
Implementación de una máquina de vector soporte para resolver el
problema OR.
</p>

### Problema XOR

XOR es un operador lógico que implementa la disyunción exclusiva y se
comporta de acuerdo a la tabla . En este caso, aún siendo un ejemplo de
sólo cuatro observaciones, su solución no es trivial aunque se puede
resolver mediante un kernel radial.

``` r
data.xor <- data.frame(x = c(-1, -1, 1, 1), 
                      y = c(-1, 1, -1, 1),
                      class = c(1, -1, -1, 1))

modelo <- ksvm(class ~ ., 
               data = data.xor, 
               type = "C-svc", 
               kernel = "rbfdot")
# table(predict(modelo), data.xor$class)
plot(modelo, data = data.xor, 
     xlim = c(-1.1, 1.1), 
     ylim = c(-1.1, 1.1))
```

<img src="graphics/svm/xor-1.pdf" alt="Implementación de una máquina de vector soporte para resolver el problema XOR." width="65%" />
<p class="caption">
Implementación de una máquina de vector soporte para resolver el
problema XOR.
</p>

Validación de los modelos SVM
-----------------------------

Como todos los modelos supervisado, las SVM dependen de las
observaciones de entrenamiento. Si cambian estas observaciones, los
parámetros del modelo pueden cambiar. En las SVM se observa ademas que
la construcción del hiperplano depende de los vectores soporte, si se
modifican los vectores soporte se modifica el hiperplano, aunque el
resto de observaciones sean las mismas.

Por este motivo, al generar un modelo SVM se debe analizar su robustez
mediante un análisis de validación cruzada. Los análisis de validación
nos permite valorar el grado de sobreajuste de nuestro modelo a los
datos. Un modelo sobreajustado clasificará muy bien las muestras de
entrenamiento, pero su exactitud será muy baja cuando pronostique nuevas
observaciones.

### Validación cruzada *leave-one-out*

La validación *leave-one-out* (validación dejando uno fuera) es una
validación muy sencilla pero que computacionalmente puede ser muy
costosa. El procedimiento de la validación *leave-one-out* se basa en lo
siguiente: Si tenemos *n* observaciones entonces construimos *n*
Máquinas de Vector Soporte,
{*S**V**M*<sub>1</sub>, *S**V**M*<sub>2</sub>, …, *S**V**M*<sub>*n*</sub>},
cada una de ellas con *n-1* muestras,
*S**V**M*<sub>*i*</sub> = {1, 2, …, *i* − 1, *i* + 1, …, *n*} y
aplicamos el modelo sobre la muestra no utilizada en la construcción del
modelo.

### Validación cruzada *k-fold*

La validación *k-fold* consiste en separar las muestras en *k* grupos de
mismo tamaño. Se construyen entonces *k* Máquinas de Vector Soporte con
*k-1* grupos y se aplica el modelo resultante sobre el grupo de muestras
no incluidas en el modelo. Observamos que si *k = n* estamos ante la
validación *leave-one-out*.

Versión
=======

Bibliografía
============

\[1\] Krzywinski M, Altman N. Points of significance: Two-factor
designs. Nature Methods. 2014;11:1187–1188.

\[2\] Altman N, Krzywinski M. Simple linear regression. Nature Methods.
2015;12:999–1000.

\[3\] Altman N, Krzywinski M. Points of significance: Analyzing
outliers: Influential or nuisance? Nature Methods. 2016;13:281–282.

\[4\] Altman N, Krzywinski M. Points of significance: Regression
diagnostics. Nature Methods. 2016;13:385–386.

\[5\] Kuehl R, Osuna M. Diseño de experimentos: Principios estadísticos
de diseño y análisis de investigación. International Thomson Editores,
S. A. de C. V. 2001.

\[6\] Pulido H, Vara Salazar R de la, González P, et al. Análisis y
diseño de experimentos. McGraw-Hill; 2004.

\[7\] Martínez-Arranz I, Mayo R, Pérez-Cormenzana M, et al. Enhancing
metabolomics research through data mining. Journal of Proteomics.
2015;127, Part B:275–288.

\[8\] Xie Y. Knitr: A comprehensive tool for reproducible research in R.
In: Stodden V, Leisch F, Peng RD, editors. Implementing reproducible
computational research \[Internet\]. Chapman; Hall/CRC; 2014. Available
from: <http://www.crcpress.com/product/isbn/9781466561595>.

\[9\] Xie Y. Dynamic documents with R and knitr \[Internet\]. 2nd ed.
Boca Raton, Florida: Chapman; Hall/CRC; 2015. Available from:
<http://yihui.name/knitr/>.

\[10\] Xie Y. Knitr: A general-purpose package for dynamic report
generation in r \[Internet\]. 2016. Available from:
<http://yihui.name/knitr/>.

\[11\] Armitage EG, Barbas C. Metabolomics in cancer biomarker
discovery: Current trends and future perspectives. J Pharm Biomed Anal.
2014;87:1–11.

\[12\] Čuperlović-Culf M. 5 - metabolomics data analysis – processing
and analysis of a dataset. In: Čuperlović-Culf M, editor. {NMR}
metabolomics in cancer research \[Internet\]. Woodhead Publishing; 2013.
pp. 261–333. Available from:
<http://www.sciencedirect.com/science/article/pii/B9781907568848500056>.

\[13\] Fox J. Applied regression analysis, linear models, and related
methods. SAGE Publications; 1997.
