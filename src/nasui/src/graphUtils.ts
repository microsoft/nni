import lodash from 'lodash';

interface Node {
  name: string,
  tail: string,
  parent: string,
  meta: SubgraphMetadata
};

class Edge {
  source: string;
  target: string;

  constructor(source: string, target: string) {
    this.source = source;
    this.target = target;
  }

  id() {
    return JSON.stringify([this.source, this.target]);
  }
};

interface Graph {
  nodes: Node[],
  edges: Edge[],
};

interface SubgraphMetadata {
  name: string,
  nodeCount: number,
  edgeCount: number,
  inputs: string[],
  outputs: string[],
  attributes: string,
  op: string
};

export function preprocessGraphData(graph: any): void {
  let cluster = Object.create({});
  graph['node'] = graph['node'].map((node: any) => {
    const split = node['name'].split('/');
    if (split.length === 1) {
      node['parent'] = '';
      node['tail'] = split[0];
    } else {
      node['parent'] = split.slice(0, -1).join('/');
      node['tail'] = split.slice(-1)[0];
      for (let i = 1; i < split.length; ++i) {
        const name = split.slice(0, i).join('/');
        const child = split.slice(0, i + 1).join('/');
        if (!cluster.hasOwnProperty(name)) {
          const parent = i > 1 ? split.slice(0, i - 1).join('/') : '';
          const tail = split[i - 1];
          cluster[name] = {
            children: [],
            parent: parent,
            tail: tail,
            name: name
          };
        }
        cluster[name].children.push(child);
      }
    }
    return node;
  });
  graph.cluster = Object.values(cluster);
  inferMutableEdges(graph);
}

function forwardPassAdjacentList(graph: any): any {
  let forwardGraph = Object.create({});
  graph['node'].forEach((node: any) => {
    if (!node.hasOwnProperty('input'))
      return;
    node['input'].forEach((input: any) => {
      if (!forwardGraph.hasOwnProperty(input))
        forwardGraph[input] = [];
      forwardGraph[input].push(node['name']);
    });
  });
  return forwardGraph;
}

function backwardPassAdjacentList(graph: any): any {
  let backwardGraph = Object.create({});
  graph.node.forEach((node: any) => {
    if (node.hasOwnProperty('input'))
      backwardGraph[node.name] = node.input;
    else
      backwardGraph[node.name] = [];
  });
  return backwardGraph;
}

function sourceNodeList(graph: any): string[] {
  let sources: string[] = [];
  graph['node'].forEach((node: any) => {
    if (node['name'].startsWith('input')) {
      sources.push(node['name'] as string);
    }
  });
  return sources;
}

export function eliminateSidechainNodes(graph: any): void {
  const adjList = forwardPassAdjacentList(graph);
  const sources = sourceNodeList(graph);
  let visitedNodes = new Set(sources);
  let dfsStack = lodash.cloneDeep(sources);
  while (dfsStack.length > 0) {
    const u = dfsStack.pop()!;
    if (adjList.hasOwnProperty(u)) {
      adjList[u].forEach((v: string) => {
        if (!visitedNodes.has(v)) {
          visitedNodes.add(v);
          dfsStack.push(v);
        }
      });
    }
  }
  graph['node'] = graph['node'].filter((node: any) => visitedNodes.has(node['name']));
}

function constructModuleTree(graph: any): any {
  let tree = Object.create({});
  const addEdge = (node: any) => {
    const parent = node['parent'] || '';
    if (!tree.hasOwnProperty(parent))
      tree[parent] = [];
    tree[parent].push(node['name']);
  };
  graph['node'].forEach(addEdge);
  graph['cluster'].forEach(addEdge);
  return tree;
}

function nodeNameToData(graph: any): any {
  let result = Object.create({});
  const addData = (node: any) => {
    result[node['name']] = node;
  }
  graph['node'].forEach(addData);
  graph['cluster'].forEach(addData);
  return result;
}

export function collapseTo(graph: any, collapsedNodes: string[]): any {
  const collapsedSet = new Set(collapsedNodes);
  const moduleTree = constructModuleTree(graph);
  const collapseResult = Object.create({});
  const dfs = (node: string, top: string | null) => {
    collapseResult[node] = top === null ? node : top;
    if (moduleTree.hasOwnProperty(node)) {
      for (const child of moduleTree[node]) {
        dfs(child, collapsedSet.has(node) && top === null ? node : top);
      }
    }
  };
  dfs('', null);
  return collapseResult;
}

export function defaultCollapsedNodes(graph: any): string[] {
  const moduleTree = constructModuleTree(graph);
  const name2data = nodeNameToData(graph);
  let result: string[] = [];
  const dfs = (node: string): number => {
    let mutableCount = 0;
    if (node === '') {
      mutableCount++;
    } else if (name2data[node]['tail'].startsWith("LayerChoice") ||
      name2data[node]['tail'].startsWith("InputChoice")) {
      mutableCount++;
    }
    if (moduleTree.hasOwnProperty(node)) {
      for (const child of moduleTree[node]) {
        mutableCount += dfs(child);
      }
    }
    if (mutableCount === 0)
      result.push(node);
    return mutableCount;
  };
  dfs('');
  return result;
}

export function subgraphMetadata(graph: any, node: string): SubgraphMetadata | null {
  const moduleTree = constructModuleTree(graph);
  const name2data = nodeNameToData(graph);
  if (!node || !name2data.hasOwnProperty(node))
    return null;
  let subgraphNodes = new Set<string>();
  const dfs = (node: string) => {
    subgraphNodes.add(node);
    if (moduleTree.hasOwnProperty(node)) {
      moduleTree[node].forEach((child: string) => { dfs(child) });
    }
  };
  dfs(node);
  let inputs: string[] = [], outputs: string[] = [];
  let domesticEdges = 0;
  for (const node of graph.node) {
    if (!node.hasOwnProperty('input'))
      continue;
    const target: string = node.name;
    for (const input of node.input) {
      if (moduleTree.hasOwnProperty('input')) {
        if (subgraphNodes.has(target) && !subgraphNodes.has(input))
          inputs.push(input);
        if (subgraphNodes.has(input) && !subgraphNodes.has(target))
          outputs.push(target);
        if (subgraphNodes.has(input) && subgraphNodes.has(target))
          domesticEdges++;
      }
    }
  }

  const attributes = name2data[node].hasOwnProperty('attr') ?
    atob(name2data[node].attr.attr.s) : '';
  const op = name2data[node].hasOwnProperty('op') ? name2data[node].op : '';
  return {
    name: node,
    nodeCount: subgraphNodes.size,
    edgeCount: domesticEdges,
    inputs: inputs,
    outputs: outputs,
    attributes: attributes,
    op: op
  }
}

function inferMutableEdges(graph: any): void {
  graph.mutableEdges = Object.create({});
  const backwardGraph = backwardPassAdjacentList(graph);
  const forwardGraph = forwardPassAdjacentList(graph);
  const name2data = nodeNameToData(graph);
  const path2module = (path: any[]) => {
    return path.map((p) => p.name ? `${p.type}[${p.name}]` : p.type).join('/');
  };
  const module2chains = (moduleName: string): Edge[][] => {
    let inputs: string[] | null = null;
    let listConstructNode: string | null = null;
    for (const node of graph.node) {
      if (node.op === 'prim::ListConstruct' && node.parent === moduleName) {
        inputs = node.input;
        listConstructNode = node.name;
        break;
      }
    }
    if (inputs === null || listConstructNode === null)
      return [];
    return inputs.map((input: string): Edge[] => {
      const visitedNodes = new Set<string>();
      let edgeSet: Edge[] = [];
      const dfs = (node: string, backward: boolean) => {
        if (visitedNodes.has(node))
          return;
        visitedNodes.add(node);
        if (!name2data[node].parent || !name2data[node].parent.startsWith(moduleName))
          return;
        const g = backward ? backwardGraph : forwardGraph;
        if (g.hasOwnProperty(node)) {
          g[node]
            .filter((to: string) => name2data.hasOwnProperty(to))
            .forEach((to: string) => {
              if (backward) {
                edgeSet.push(new Edge(to, node));
              } else {
                edgeSet.push(new Edge(node, to));
              }
              dfs(to, backward);
            });
        }
      };
      edgeSet.push(new Edge(input, listConstructNode!));
      dfs(input, true);
      visitedNodes.clear();
      dfs(listConstructNode!, false);
      return edgeSet;
    });
  }
  Object.entries(graph.mutable).forEach((obj) => {
    const [key, paths] = obj;
    const modules = (paths as any[]).map(path2module);
    const moduleEdge = modules.map(module2chains);
    let edges: Edge[][] = [];
    for (let i = 0; ; ++i) {
      if (moduleEdge.filter((me) => i < me.length).length === 0)
        break;
      edges.push([]);
      moduleEdge
        .filter((me) => i < me.length)
        .forEach((me) => {
          edges[i].push(...me[i]);
        })
    }
    graph.mutableEdges[key] = edges;
  });
}
