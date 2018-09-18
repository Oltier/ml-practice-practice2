### STUDENT TASK ###
# x=...
# y=...
# YOUR CODE HERE
raise NotImplementedError()
plt.figure(figsize=(8, 8))
plt.scatter(x, y)
plt.title(r'$\bf{Figure\ 3.}$Normalized cryptocurrency prices ($x^{(i)},y^{(i)}$)')
plt.xlabel('normalized Bitcoin prices')
plt.ylabel('normalized Ethereum prices')
plt.annotate('$(x^{(i)},y^{(i)})$', xy=(x[913], y[913]), xytext=(0.1, 0.5),
             arrowprops=dict(arrowstyle="->", facecolor='black'),
             )
axis = plt.gca()

# Check that plot is a scatterplot. Requires plotchecker
from plotchecker import ScatterPlotChecker

pc = ScatterPlotChecker(axis)
assert len(pc.x_data) == 948
assert len(pc.y_data) == 948
