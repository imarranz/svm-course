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
       title = "dígito")
