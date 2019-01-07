## --- librer?as ----

library("caret")
library("e1071")
library("kernlab")

## ---- example ----

set.seed(123)

x1 <- runif(1000, min = 0, max = pi)
y1 <- sin(x1) + rnorm(1000, sd = 0.2)

x2 <- runif(1000, min = pi, max = 2*pi)
y2 <- sin(x2)
x2 <- x2 - pi/2
y2 <- y2 + 0.5 + rnorm(1000, sd = 0.2)

ESVM <- cbind(x1 = c(x1, x2), 
              x2 = c(y1, y2), 
              y = rep(c(1,-1), each = 1000))
ESVM <- as.data.frame(ESVM)

modelo <- ksvm(y ~ ., 
               data = ESVM[, c(2,1,3)], 
               kernel = "rbfdot", 
               type = "C-svc", 
               kpar = list(sigma = 0.097), 
               C = 9.74)
               
plot(modelo, data = ESVM[, 1:2], grid = 50)               

## ---- load data ----

load(file = "E:/valores.Rdata")

print(head(entrenamiento[, 1:10]))
print(head(validacion[, 1:10]))
print(head(prueba[, 1:10]))

## ---- PCA ----

mipc <- prcomp(entrenamiento[, 1:256])
mipch <- c(21:25,21:25)
plot(predict(mipc)[, 1:2], 
     xlim = c(-10, 10),
     ylim = c(-10, 10),
     pch = rep(mipch, each = 65),
     col = gray(0.1), #rep(rainbow(10), each = 65),
     bg = rep(rainbow(10), each = 65),
     xaxt = "n", yaxt = "n",
     xlab = "Componente Principal 1",
     ylab = "Componente Principal 2")
axis(1)
axis(2, las = 2)
abline(v = pretty(c(-10, 10)), lty = 3, col = "gray")
abline(h = pretty(c(-10, 10)), lty = 3, col = "gray")

legend(x = "topright", legend = seq(0, 9), pch = c(21:25,21:25), 
       col = gray(0.1), #rainbow(10),
       pt.bg = rainbow(10), bty = "n",
       ncol = 5,cex = 0.7,
       title = "d?gito")

## ---- modelo_svmlinear2_paso1 ----

X <- entrenamiento[, 1:256]
Y <- entrenamiento$digito

head(X[, 1:10])

# Modelo lineal
modelo <- train(x = X,
                y = Y,
                method = "svmLinear2")

# Resultados del modelo
print(modelo)

# Clasificaci?n
table(Pronostico = predict(modelo), 
      Referencia = rep(seq(0, 9, 1), each = 65))

# Clasificaci?n en las muestras de validaci?n
pron <- predict(modelo, newdata = validacion[, 1:256])
refe <- rep(seq(0, 9, 1), each = 65)  

table(Pronostico = pron,
      Referencia = refe
)

# Muestras de validaci?n falladas
filtro <- pron != refe
predicciones_fallidas <- pron[filtro]
failcases <- strsplit(rownames(validacion)[filtro], "_")
failcases

# Representaci?n gr?fica de las muestras de validaci?n falladas
par(mfrow = c(6, 3), mar = c(0.2,0.2,0.2,0.2))
for (i in 1:18) {
  plot(-1, -1, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
  text(x = 0.5, y = 0.5, family = failcases[[i]][1], failcases[[i]][2], cex = 5)      
  text(x = 0.1, y = 0.1, predicciones_fallidas[i], cex = 2, col = "gray")      
}

# Variables m?s importantes
a <- varImp(modelo)
colnames(a$importance) <- paste("d?gito", 
                                seq(0, 9), 
                                sep = " ")
plot(a, 
     top = 20, 
     xlab = "Variables (p?xeles) m?s importantes")

## ---- modelo_svmlinear2_paso2 ----

X <- entrenamiento[, 1:256]
Y <- entrenamiento$digito

head(X[, 1:10])

# Modelo lineal con par?metros
modelo <- train(x = X,
                y = Y,
                preProcess = c("scale", "center"),
                trControl = trainControl(method = "cv",
                                         p = 0.8, 
                                         number = 5,
                                         repeats = 5),
                method = "svmLinear2",
                tuneLength = 10,
                maximize = TRUE,
                metric = "Accuracy")     

# Resultados del modelo
print(modelo) 

# Clasificación
table(Pronostico = predict(modelo), 
      Referencia = rep(seq(0, 9, 1), each = 65))

# Clasificación en las muestras de validaci?n
pron <- predict(modelo, newdata = validacion[, 1:256])
refe <- rep(seq(0, 9, 1), each = 65)  

table(Pronostico = pron,
      Referencia = refe
)

# Muestras de validaci?n falladas
filtro <- pron != refe
predicciones_fallidas <- pron[filtro]
failcases <- strsplit(rownames(validacion)[filtro], "_")
failcases

# Representaci?n gr?fica de las muestras de validaci?n falladas
par(mfrow = c(7, 4), mar = c(0.2,0.2,0.2,0.2))
for (i in 1:28) {
  plot(-1, -1, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
  text(x = 0.5, y = 0.5, family = failcases[[i]][1], failcases[[i]][2], cex = 5)      
  text(x = 0.1, y = 0.1, predicciones_fallidas[i], cex = 2, col = "gray")      
}

# Variables m?s importantes
a <- varImp(modelo)
colnames(a$importance) <- paste("d?gito", 
                                seq(0, 9), 
                                sep = " ")
plot(a, 
     top = 20, 
     xlab = "Variables (p?xeles) m?s importantes")

## ----  ----

X <- prueba[, 1:256]
Y <- prueba$digito

procesado <- list(NULL, c("scale", "center"), "range")

modelo <- vector(mode = "list", length = 3)

for (i1 in 1:length(procesado)) {
  modelo[[i1]] <- train(x = X,
                  y = Y,
                  preProcess = procesado[[i1]],
                  tuneLength = 5,
                  method = "svmLinear2")     
}

lapply(modelo, FUN = function(mod) {mod$results})

for(i1 in 1:length(procesado)) {
  pron <- predict(modelo[[i1]], newdata = validacion[, 1:256])
  refe <- rep(seq(0, 9, 1), each = 65)  
  print(sum(diag(table(pron, refe)))/650)
}

