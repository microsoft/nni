function path2module(path: any[]): string {
  return path.map(p => p.name ? `${p.type}[${p.name}]` : p.type).join('/');
}

function opName(rawName: string): string {
  if (rawName.includes('::')) {
    return rawName.split('::')[1];
  } else {
    return rawName;
  }
}

export class NodeTs {
  readonly id: string;
  readonly tail: string;
  parent: NodeTs | undefined;
  children: NodeTs[];
  op: string;
  attributes: string;

  constructor(id: string, tail: string, op: string, attributes: string) {
    this.children = [];
    this.id = id;
    this.tail = tail;
    this.parent = undefined;
    this.op = op;
    this.attributes = attributes;
  }

  descendants(leafOnly: boolean): NodeTs[] {
    // return all descendants includinng itself
    const result: NodeTs[] = [];
    if (!leafOnly || this.isChildless())
      result.push(this);
    for (const child of this.children) {
      const childDesc = child.descendants(leafOnly);
      if (childDesc.length > 0)
        result.push(...childDesc);
    }
    return result;
  }

  isChildless(): boolean {
    return this.children.length === 0;
  }

  isParent(): boolean {
    return this.children.length > 0;
  }

  toString(): string {
    return `Node(id=${this.id})`;
  }
};

export class Edge {
  readonly source: NodeTs;
  readonly target: NodeTs;
  readonly id: string;

  constructor(source: NodeTs, target: NodeTs) {
    this.source = source;
    this.target = target;
    this.id = JSON.stringify([this.source.id, this.target.id]);
  }

  toString(): string {
    return `Edge(${this.source} -> ${this.target})`;
  }
};

interface NodeSummary {
  name: string,
  nodeCount: number,
  edgeCount: number,
  inputs: string[],
  outputs: string[],
  attributes: string,
  op: string
};

export class Graph {
  root: NodeTs;
  nodes: NodeTs[];
  edges: Edge[];
  defaultExpandSet: Set<string>;
  mutableEdges: Map<string, Edge[][]>;
  private id2idx: Map<string, number>;
  private edgeId2idx: Map<string, number>;
  private forwardGraph: Map<string, string[]>;
  private backwardGraph: Map<string, string[]>;
  private node2edge: Map<string, Edge[]>;

  private build() {
    this.id2idx.clear();
    this.nodes.forEach((node, i) => {
      this.id2idx.set(node.id, i);
    });
    this.edgeId2idx.clear();
    this.edges.forEach((edge, i) => {
      this.edgeId2idx.set(edge.id, i);
    });

    this.forwardGraph.clear();
    this.backwardGraph.clear();
    this.node2edge.clear();
    this.edges.forEach(edge => {
      if (!this.forwardGraph.has(edge.source.id))
        this.forwardGraph.set(edge.source.id, []);
      this.forwardGraph.get(edge.source.id)!.push(edge.target.id);
      if (!this.backwardGraph.has(edge.target.id))
        this.backwardGraph.set(edge.target.id, []);
      this.backwardGraph.get(edge.target.id)!.push(edge.source.id);
      if (!this.node2edge.has(edge.source.id))
        this.node2edge.set(edge.source.id, []);
      if (!this.node2edge.has(edge.target.id))
        this.node2edge.set(edge.target.id, []);
      this.node2edge.get(edge.source.id)!.push(edge);
      this.node2edge.get(edge.target.id)!.push(edge);
    });

    this.root.children = this.nodes.filter(node => node.parent === undefined);
    // won't set parent for these nodes, leave them as undefined
    this.nodes.forEach(node => {
      node.children = node.children.filter(child => this.getNodeById(child.id) !== undefined);
    });
  }

  getNodeById(id: string): NodeTs | undefined {
    const idx = this.id2idx.get(id);
    if (idx === undefined) return undefined;
    return this.nodes[idx];
  }

  getEdgeById(source: string, target: string): Edge | undefined {
    const idx = this.edgeId2idx.get(JSON.stringify([source, target]));
    if (idx === undefined) return undefined;
    return this.edges[idx];
  }

  constructor(graphData: any, eliminateSidechains: boolean) {
    this.id2idx = new Map<string, number>();
    this.edgeId2idx = new Map<string, number>();
    this.forwardGraph = new Map<string, string[]>();
    this.backwardGraph = new Map<string, string[]>();
    this.node2edge = new Map<string, Edge[]>();
    this.root = new NodeTs('', '', '', '');
    const cluster = new Map<string, NodeTs>();
    const parentMap = new Map<string, string>();
    this.nodes = graphData.node.map((node: any): NodeTs => {
      const split = node.name.split('/');
      const attr = node.hasOwnProperty('attr') ? atob(node.attr.attr.s) : '';
      if (split.length === 1) {
        return new NodeTs(node.name, node.name, opName(node.op), attr);
      } else {
        parentMap.set(node.name, split.slice(0, -1).join('/'));
        // create clusters
        for (let i = 1; i < split.length; ++i) {
          const name = split.slice(0, i).join('/');
          if (!cluster.has(name)) {
            const parent = i > 1 ? split.slice(0, i - 1).join('/') : '';
            const tail = split[i - 1];
            cluster.set(name, new NodeTs(name, tail, '', ''));
            parentMap.set(name, parent);
          }
        }
        return new NodeTs(node.name, split.slice(-1)[0], opName(node.op), attr);
      }
    });
    cluster.forEach(node => this.nodes.push(node));
    this.nodes.forEach((node, i) => {
      this.id2idx.set(node.id, i);
    });
    parentMap.forEach((parent, child) => {
      const [childNode, parentNode] = [child, parent].map(this.getNodeById.bind(this));
      if (childNode !== undefined && parentNode !== undefined) {
        childNode.parent = parentNode;
        parentNode.children.push(childNode);
      }
    });

    // build edges
    this.edges = [];
    graphData.node.forEach((node: any) => {
      if (!node.hasOwnProperty('input')) return;
      const target = this.getNodeById(node.name);
      if (target === undefined) return;
      node.input.forEach((input: string) => {
        const source = this.getNodeById(input);
        if (source !== undefined) {
          this.edges.push(new Edge(source, target));
        }
      })
    })
    this.build();

    if (eliminateSidechains) {
      this.eliminateSidechains();
    }
    this.defaultExpandSet = this.getDefaultExpandSet(graphData.mutable);
    this.mutableEdges = this.inferMutableEdges(graphData.mutable);
  }

  private eliminateSidechains(): void {
    const sources = this.nodes
      .map(node => node.id)
      .filter(id => id.startsWith('input'));

    const visitedNodes = new Set(sources);
    const dfsStack = sources;
    while (dfsStack.length > 0) {
      const u = dfsStack.pop()!;
      if (this.forwardGraph.has(u)) {
        this.forwardGraph.get(u)!.forEach((v: string) => {
          if (!visitedNodes.has(v)) {
            visitedNodes.add(v);
            dfsStack.push(v);
          }
        });
      }
    }
    const compoundCheck = (node: NodeTs) => {
      if (node.isChildless())
        return visitedNodes.has(node.id);
      for (const child of node.children)
        if (compoundCheck(child))
          visitedNodes.add(node.id);
      return visitedNodes.has(node.id);
    }
    compoundCheck(this.root);
    this.nodes = this.nodes.filter(node => visitedNodes.has(node.id));
    this.edges = this.edges.filter(edge =>
      visitedNodes.has(edge.source.id) && visitedNodes.has(edge.target.id));
    this.build();
  }

  private getDefaultExpandSet(graphDataMutable: any): Set<string> {
    // if multiple, only expand first
    const whitelistModuleList = Object.values(graphDataMutable)
      .filter(Boolean)
      .map((paths: any) => path2module(paths[0]));
    const whitelistModule = new Set(whitelistModuleList);

    const result = new Set<string>();
    const dfs = (node: NodeTs): number => {
      // node with mutableCount greater than 0 won't be collapsed
      let mutableCount = 0;
      if (node.id === '') {
        // root node
        mutableCount++;
      } else if (whitelistModule.has(node.id)) {
        mutableCount++;
      } else if (node.parent !== undefined && whitelistModule.has(node.parent.id)) {
        mutableCount++;
      }
      mutableCount += node.children.map(child => dfs(child)).reduce((a, b) => a + b, 0);
      if (mutableCount > 0 && node.isParent())
        result.add(node.id);
      return mutableCount;
    };
    dfs(this.root);
    return result;
  }

  private inferMutableModule(moduleName: string): Edge[][] {
    let inputs: string[] | undefined = undefined;
    let listConstructNode: string | undefined = undefined;
    const moduleNode = this.getNodeById(moduleName);
    if (moduleNode === undefined) return [];
    for (const node of moduleNode.children)
      if (node.op === 'ListConstruct') {
        inputs = this.backwardGraph.get(node.id);
        listConstructNode = node.id;
        break;
      }
    if (inputs === undefined || listConstructNode === undefined)
      return [];
    return inputs.map((input: string): Edge[] => {
      const visitedNodes = new Set<string>();
      const edgeSet: Edge[] = [];
      const dfs = (node: string, backward: boolean) => {
        const nodeData = this.getNodeById(node)!;
        if (visitedNodes.has(node)) return;
        visitedNodes.add(node);
        if (nodeData.parent === undefined || !nodeData.parent.id.startsWith(moduleName)) {
          // in another module now
          return;
        }
        const g = backward ? this.backwardGraph : this.forwardGraph;
        const glist = g.get(node);
        if (glist !== undefined) {
          glist
            .forEach((to: string) => {
              edgeSet.push(backward ?
                this.getEdgeById(to, node)! :
                this.getEdgeById(node, to)!);
              dfs(to, backward);
            });
        }
      };
      edgeSet.push(this.getEdgeById(input, listConstructNode!)!);
      dfs(input, true);
      visitedNodes.clear();
      dfs(listConstructNode!, false);
      return edgeSet;
    });
  }

  private inferMutableEdges(graphDataMutable: any): Map<string, Edge[][]> {
    const result = new Map<string, Edge[][]>();
    Object.entries(graphDataMutable).forEach(obj => {
      const [key, paths] = obj;
      const modules = (paths as any[]).map(path2module);
      const moduleEdge = modules.map(this.inferMutableModule.bind(this));
      const edges: Edge[][] = [];
      for (let i = 0; ; ++i) {
        if (moduleEdge.filter(me => i < me.length).length === 0)
          break;
        edges.push([]);
        moduleEdge
          .filter(me => i < me.length)
          .forEach(me => edges[i].push(...me[i]));
      }
      result.set(key, edges);
    });
    return result;
  }

  private connectedEdges(node: string): Edge[] {
    const result = this.node2edge.get(node);
    if (result === undefined)
      return [];
    return result;
  }

  nodeSummary(node: NodeTs | string): NodeSummary | undefined {
    if (typeof node === 'string') {
      const nodeData = this.getNodeById(node);
      if (nodeData === undefined) return undefined;
      return this.nodeSummary(nodeData);
    }
    const descendants = node.descendants(false);
    const descendantSet = new Set(descendants.map(node => node.id));
    const inputs = new Set<string>();
    const outputs = new Set<string>();
    let domesticEdges = 0;
    for (const edge of this.edges) {
      const [source, target] = [edge.source.id, edge.target.id];
      if (descendantSet.has(target) && !descendantSet.has(source))
        inputs.add(source);
      if (descendantSet.has(source) && !descendantSet.has(target))
        outputs.add(target);
      if (descendantSet.has(source) && descendantSet.has(target))
        domesticEdges++;
    }
    return {
      name: node.id,
      nodeCount: descendants.length,
      edgeCount: domesticEdges,
      inputs: Array.from(inputs),
      outputs: Array.from(outputs),
      attributes: node.attributes,
      op: node.op
    }
  }

  weightFromMutables(mutable: any): Map<string, number> {
    const elemWeight = new Map<string, number>();
    Object.entries(mutable).forEach(entry => {
      const elemWeightPartial = new Map<string, number>();
      const key = entry[0];
      const weights = entry[1] as number[];
      this.mutableEdges.get(key)!.forEach((edges: any, i: number) => {
        edges.forEach((edge: any) => {
          if (elemWeightPartial.has(edge.id)) {
            elemWeightPartial.set(edge.id, elemWeightPartial.get(edge.id)! + weights[i]);
          } else {
            elemWeightPartial.set(edge.id, weights[i]);
          }
        })
      });
      elemWeightPartial.forEach((v, k) => {
        if (elemWeight.has(k)) {
          elemWeight.set(k, Math.min(elemWeight.get(k)!, v));
        } else {
          elemWeight.set(k, v);
        }
      });
    });
    this.nodes.forEach(node => {
      const edges = this.connectedEdges(node.id);
      const relatedEdges = edges.filter(edge => elemWeight.has(edge.id));
      if (relatedEdges.length > 0) {
        if (relatedEdges.length < edges.length) {
          elemWeight.set(node.id, 1.);
        } else {
          // all related edge
          const nw = edges.map(edge => elemWeight.get(edge.id)!)
            .reduce((a, b) => Math.max(a, b));
          elemWeight.set(node.id, nw);
        }
      }
    });
    elemWeight.forEach((v, k) => elemWeight.set(k, Math.min(v, 1.)));

    // set compound weight
    const gatherWeightsFromChildren = (node: NodeTs): number | undefined => {
      if (node.isParent()) {
        const childrenWeights =
          node.children.map(gatherWeightsFromChildren)
            .filter(val => val !== undefined);
        if (childrenWeights.length > 0) {
          const nw = childrenWeights.reduce((a, b) => Math.max(a!, b!));
          elemWeight.set(node.id, nw!);
          return nw;
        } else {
          return undefined;
        }
      } else {
        return elemWeight.get(node.id);
      }
    };
    gatherWeightsFromChildren(this.root);
    return elemWeight;
  }
};
