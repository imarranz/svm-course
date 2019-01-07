# http://stackoverflow.com/questions/7121807/how-to-change-the-font-of-the-main-title-in-plot
# http://blog.revolutionanalytics.com/2012/09/how-to-use-your-favorite-fonts-in-r-charts.html

library("extrafont")
# font_import()
# fonts()
# fonttable() 

fuentes <- fonts()

for (i in 1:length(fonts())) {
  for (j in 0:9) {
    filename <- paste("/home/ibon/Escritorio/svm/ejercicio/numeros/", j, "/", fuentes[i], ".png", sep = "")
    png(filename = filename, width = 30, height = 30)
    par(mfrow = c(1, 1), mar = c(0,0,0,0))
    plot(-1, -1, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
    text(x = 0.5, y = 0.5, family = fuentes[i], j, cex = 3)    
    dev.off()
  }
}


j <- 8
par(mfrow = c(13, 10), mar = c(0.1, 0.1, 1, 0.1))
for (i in 1:130) {
  plot(-1, -1, xlim = c(0, 1), ylim = c(0, 1), axes = FALSE)
  text(x = 0.5, y = 0.5, family = fuentes[i], j, cex = 3)    
  mtext(text = substring(fuentes[i], 1, 12), side = 3, cex = 0.6, col = gray(0.4))
}