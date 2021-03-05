import numpy as np

h_states = 3
v_states = 3
q_table = np.zeros([h_states*v_states,2])
q_table = np.array([[0.0, -1.0], [2.0, 3.0], [4.0, 5.0],
                    [6.0, 7.0], [8.0, 9.0], [10.0, 11.0],
                    [12.0, 13.0], [14.0, 15.0], [16.0, 17.0],
                    ])


print(q_table)
print("\n")
select = [1*v_states + 2, 1*v_states + 1]
print(q_table[select])
best_actions = np.argmax(q_table[select],axis=1)
print("best_actions:",best_actions)
print(q_table[select,best_actions])
print("")
q_table[select,best_actions] = (q_table[select,best_actions]+3)*3
print(q_table)
