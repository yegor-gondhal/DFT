import basis_set_exchange as bse
import json

basis = bse.get_basis("cc-pVDZ")

basis_functions = {}

for Z, atom in basis["elements"].items():
    atom_list = []
    for shell in atom["electron_shells"]:
        l = shell["angular_momentum"]
        exponents = [float(x) for x in shell["exponents"]]
        coefficients = [
                [float(c) for c in row]
                for row in shell["coefficients"]
            ]
        atom_list.append([l, exponents, coefficients])

    basis_functions[f"{Z}"] = atom_list


with open("data1.json", "w") as file:
    json.dump(basis_functions, file, indent=4)
