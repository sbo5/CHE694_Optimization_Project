from casadi import nlpsol
dv = 1
J = 1
g = 1
x0, lbx, ubx, lbg, ubg = 1,1,1,1,


# Build solver
nlp = {'x': dv, 'f': J, 'g': g}
solver = nlpsol('solver', 'ipopt', nlp)

# Call solver
solution = solver(x0, lbx, ubx, lbg, ubg)