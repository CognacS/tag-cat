from ..generators import DidacticSamplesGenerator, GeneratorsCollection

def main():
    structure = 'X*Z*+Z*X*+Y='
    
    values = {'X':['9'], 'Z':['0'], 'Y':['0', '1']}
    gen1 = DidacticSamplesGenerator(structure, values)

    structure = 'X*+Y='
    values = {'X':['9'], 'Y':['0', '1']}
    gen2 = DidacticSamplesGenerator(structure, values)

    g_col = GeneratorsCollection([gen1, gen2])
    print(g_col([2,4], [4, 11]))
    print(g_col([2,4], [4, 11]))
    print(g_col([2,4], [4, 11]))

if __name__ == '__main__':
    main()