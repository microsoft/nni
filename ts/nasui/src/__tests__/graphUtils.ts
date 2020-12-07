import { Graph } from '../graphUtils';
import *  as fs from 'fs';

describe('Graph utils test', () => {
  it('Graph constructor darts', () => {
    const graph = new Graph(JSON.parse(fs.readFileSync('assets/darts/graph.json').toString()), true);
    const activation = JSON.parse(fs.readFileSync('assets/darts/log').toString().split('\n')[0]);
    expect(graph.nodes.length).toEqual(1842);
    expect(graph.edges.length).toEqual(927);
    const weights = graph.weightFromMutables(activation);
    expect(weights.get('["CNN/ModuleList[cells]/Cell[1]/ModuleList[mutable_ops]/Node[0]/InputChoice[input_switch]/input.228",' +
                       '"CNN/ModuleList[cells]/Cell[1]/ModuleList[mutable_ops]/Node[1]/ModuleList[ops]/' + 
                       'LayerChoice[2]/PoolBN[maxpool]/MaxPool2d[pool]/input.229"]')).toBeCloseTo(0.125, 3);
  });

  it('Graph constructor naive', () => {
    const graph = new Graph(JSON.parse(fs.readFileSync('assets/naive/graph.json').toString()), true);
    expect(graph.nodes.length).toEqual(51);
    expect(graph.edges.length).toEqual(37);
    expect(graph.mutableEdges.get('LayerChoice1')![0].length).toEqual(5);
    expect(graph.mutableEdges.get('LayerChoice1')![1].length).toEqual(5);
    expect(graph.mutableEdges.get('LayerChoice2')![0].length).toEqual(5);
    expect(graph.mutableEdges.get('LayerChoice2')![1].length).toEqual(5);
    expect(graph.mutableEdges.get('InputChoice3')![0].length).toEqual(4);
  });
});
