---
title: "Untitled"
author: "Ibon Martínez Arranz"
date: "25 de octubre de 2016"
output: 
  html_document:
    toc: true
    toc_float: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


## Support vector machine

In machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier. An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall on.

In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.

When data are not labeled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering of the data to groups, and then map new data to these formed groups. The clustering algorithm which provides an improvement to the support vector machines is called support vector clustering[2] and is often used in industrial applications either when data is not labeled or when only some data is labeled as a preprocessing for a classification pass.

Contents

    1 Motivation
    2 Definition
    3 Applications
    4 History
    5 Linear SVM
        5.1 Hard-margin
        5.2 Soft-margin
    6 Nonlinear classification
    7 Computing the SVM classifier
        7.1 Primal
        7.2 Dual
        7.3 Kernel trick
        7.4 Modern methods
            7.4.1 Sub-gradient descent
            7.4.2 Coordinate descent
    8 Empirical risk minimization
        8.1 Risk minimization
        8.2 Regularization and stability
        8.3 SVM and the hinge loss
            8.3.1 Target functions
    9 Properties
        9.1 Parameter selection
        9.2 Issues
    10 Extensions
        10.1 Support vector clustering (SVC)
        10.2 Multiclass SVM
        10.3 Transductive support vector machines
        10.4 Structured SVM
        10.5 Regression
    11 Implementation
    12 See also
    13 References
    14 Bibliography
    15 External links

### Motivation

Classifying data is a common task in machine learning. Suppose some given data points each belong to one of two classes, and the goal is to decide which class a new data point will be in. In the case of support vector machines, a data point is viewed as a $p$-dimensional vector (a list of $p$ numbers), and we want to know whether we can separate such points with a $(p-1)$-dimensional hyperplane. This is called a linear classifier. There are many hyperplanes that might classify the data. One reasonable choice as the best hyperplane is the one that represents the largest separation, or margin, between the two classes. So we choose the hyperplane so that the distance from it to the nearest data point on each side is maximized. If such a hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier it defines is known as a maximum margin classifier; or equivalently, the perceptron of optimal stability.

### Definition

More formally, a support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.
Kernel machine

Whereas the original problem may be stated in a finite dimensional space, it often happens that the sets to discriminate are not linearly separable in that space. For this reason, it was proposed that the original finite-dimensional space be mapped into a much higher-dimensional space, presumably making the separation easier in that space. To keep the computational load reasonable, the mappings used by SVM schemes are designed to ensure that dot products may be computed easily in terms of the variables in the original space, by defining them in terms of a kernel function ${\displaystyle k(x,y)}$ selected to suit the problem. The hyperplanes in the higher-dimensional space are defined as the set of points whose dot product with a vector in that space is constant. The vectors defining the hyperplanes can be chosen to be linear combinations with parameters ${\displaystyle \alpha _{i}} \alpha _{i}$ of images of feature vectors ${\displaystyle x_{i}}$ that occur in the data base. With this choice of a hyperplane, the points ${\displaystyle x}$ in the feature space that are mapped into the hyperplane are defined by the relation: ${\displaystyle \textstyle \sum _{i}\alpha _{i}k(x_{i},x)=\mathrm {constant}.}$ $\textstyle \sum _{i}\alpha _{i}k(x_{i},x)=\mathrm {constant}$. Note that if ${\displaystyle k(x,y)}$ becomes small as ${\displaystyle y}$ grows further away from ${\displaystyle x}$, each term in the sum measures the degree of closeness of the test point ${\displaystyle x}$ to the corresponding data base point ${\displaystyle x_{i}} x_{i}$. In this way, the sum of kernels above can be used to measure the relative nearness of each test point to the data points originating in one or the other of the sets to be discriminated. Note the fact that the set of points ${\displaystyle x}$ mapped into any hyperplane can be quite convoluted as a result, allowing much more complex discrimination between sets which are not convex at all in the original space.

### Applications

SVMs can be used to solve various real world problems of Uncertainty in Knowledge-Based Systems (2014).

* SVMs are helpful in text and hypertext categorization as their application can significantly reduce the need for labeled training instances in both the standard inductive and transductive settings.

* Classification of images can also be performed using SVMs. Experimental results show that SVMs achieve significantly higher search accuracy than traditional query refinement schemes after just three to four rounds of relevance feedback. This is also true of image segmentation systems, including those using a modified version SVM that uses the privileged approach as suggested by Vapnik.

* Hand-written characters can be recognized using SVM.

* The SVM algorithm has been widely applied in the biological and other sciences. They have been used to classify proteins with up to 90% of the compounds classified correctly. Permutation tests based on SVM weights have been suggested as a mechanism for interpretation of SVM models. Support vector machine weights have also been used to interpret SVM models in the past.[8] Posthoc interpretation of support vector machine models in order to identify features used by the model to make predictions is a relatively new area of research with special significance in the biological sciences.

### History

The original SVM algorithm was invented by Vladimir N. Vapnik and Alexey Ya. Chervonenkis in 1963. In 1992, Bernhard E. Boser, Isabelle M. Guyon and Vladimir N. Vapnik suggested a way to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes. The current standard incarnation (soft margin) was proposed by Corinna Cortes and Vapnik in 1993 and published in 1995.


### Linear SVM

We are given a training dataset of ${\displaystyle n}$ points of the form

\begin{equation}
({\vec {x}}_{1},y_{1}),\,\ldots ,\,({\vec {x}}_{n},y_{n})
\end{equation}

where the ${\displaystyle y_{i}}$ are either 1 or ???1, each indicating the class to which the point ${\displaystyle {\vec {x}}_{i}}$ belongs. Each ${\displaystyle {\vec {x}}_{i}}$ is a ${\displaystyle p}$-dimensional real vector. We want to find the ''maximum-margin hyperplane´´ that divides the group of points ${\displaystyle {\vec {x}}_{i}}$ for which ${\displaystyle y_{i}=1}$ from the group of points for which ${\displaystyle y_{i} = -1}$, which is defined so that the distance between the hyperplane and the nearest point ${\displaystyle {\vec {x}}_{i}}$ from either group is maximized.

Any hyperplane can be written as the set of points ${\displaystyle {\vec {x}}}$ satisfying

\begin{equation}
{\displaystyle {\vec {w}}\cdot {\vec {x}} -b = 0,\,}
\end{equation}

where ${\displaystyle {\vec {w}}}$ is the (not necessarily normalized) normal vector to the hyperplane. The parameter ${\displaystyle {\tfrac {b}{\|{\vec {w}}\|}}}$ determines the offset of the hyperplane from the origin along the normal vector ${\displaystyle {\vec {w}}}$.


#### Hard-margin

If the training data are linearly separable, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The region bounded by these two hyperplanes is called the ''margin´´, and the maximum-margin hyperplane is the hyperplane that lies halfway between them. These hyperplanes can be described by the equations

\begin{equation}
{\displaystyle {\vec {w}}\cdot {\vec {x}}-b=1\,}
\end{equation}

and

\begin{equation}
{\displaystyle {\vec {w}}\cdot {\vec {x}}-b=-1.\,}
\end{equation}

Geometrically, the distance between these two hyperplanes is ${\displaystyle {\tfrac {2}{\|{\vec {w}}\|}}}$, so to maximize the distance between the planes we want to minimize ${\displaystyle \|{\vec {w}}\|}$. As we also have to prevent data points from falling into the margin, we add the following constraint: for each ${\displaystyle i}$ either

\begin{equation}
{\displaystyle {\vec {w}}\cdot {\vec {x}}_{i}-b\geq 1,} if y i = 1 {\displaystyle y_{i}=1}
\end{equation}

or

\begin{equation}
{\displaystyle {\vec {w}}\cdot {\vec {x}}_{i}-b\leq -1,} if y i = ??? 1. {\displaystyle y_{i}=-1.}
\end{equation}

These constraints state that each data point must lie on the correct side of the margin.

This can be rewritten as:

\begin{equation}
{\displaystyle y_{i}({\vec {w}}\cdot {\vec {x}}_{i}-b)\geq 1,\quad {\text{ for all }}1\leq i\leq n.\qquad \qquad (1)}
\end{equation}
