import React, { createRef } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import lodash from 'lodash';
import { collapseTo, defaultCollapsedNodes } from './graphUtils';

cytoscape.use(dagre);

type ChartProps = {
  width: number,
  height: number,
  displayStep: number,
  graphData: any,
  logData: any[],
  handleSelectionChange: (_: string) => void,
}

const styles = [
  {
    selector: 'node',
    style: {
      'shape': 'round-rectangle',
      'width': 'label',
      'height': 'label',
      'content': 'data(label)',
      'text-wrap': 'wrap',
      'text-valign': 'center',
      'text-halign': 'center',
      'font-family': '"Roboto", "Helvetica", "Arial", sans-serif',
      'font-size': 20,
      'padding-left': '8px',
      'padding-right': '8px',
      'padding-top': '8px',
      'padding-bottom': '8px',
      'background-color': '#000',
      'background-opacity': 0.05,
      'border-color': '#aaa',
      'border-width': '2px',
      'selection-box-opacity': 0,
      'active-bg-opacity': 0,
    }
  },
  {
    selector: 'node.op',
    style: {
      'background-color': '#fafafa',
      'background-opacity': 1,
    }
  },
  {
    selector: 'node:selected',
    style: {
      'border-width': '4px',
      'border-color': '#333',
    }
  },
  {
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'line-color': '#aaa',
      'target-arrow-color': '#aaa'
    }
  },
  {
    selector: 'node.compound',
    style: {
      'shape': 'roundrectangle',
      'text-valign': 'top',
      'padding-top': '30px',
      'padding-bottom': '10px',
      'padding-right': '10px',
      'padding-left': '10px',
    }
  }
];

export default class Chart extends React.Component<ChartProps> {
  container = createRef<HTMLDivElement>();
  zoom: any;
  collapseInstance: any;
  cyInstance: cytoscape.Core | null = null;
  collapsedNodes: any = Object.create({});
  elemWeight: any = Object.create({});

  componentDidMount() {
    this.cyInstance = cytoscape({
      container: this.container.current,
      elements: [],
      style: styles as any,
      wheelSensitivity: 0.1,
    });
    let singleClickedNodes = new Set<string>();
    this.cyInstance!.off('click');
    this.cyInstance!.on('click', (e: cytoscape.EventObject) => {
      if (e.target.id) {
        let nodeId = e.target.id();
        if (singleClickedNodes.has(nodeId)) {
          singleClickedNodes.delete(nodeId);
          e.target.trigger('dblclick');
        } else {
          singleClickedNodes.add(nodeId);
          setTimeout(() => { singleClickedNodes.delete(nodeId); }, 300);
        }
      }
    });
    this.cyInstance!.on('dblclick', (e: cytoscape.EventObject) => {
      let nodeId = e.target.id();
      if (this.collapsedNodes[nodeId]) {
        this.collapsedNodes[nodeId] = false;
      } else {
        this.collapsedNodes[nodeId] = true;
      }
      this.renderGraph(false);
    });
    this.cyInstance!.on('select', (e: cytoscape.EventObject) => {
      this.props.handleSelectionChange(e.target.id());
    });
    this.renderGraph(true);
  }

  componentDidUpdate(prevProps: ChartProps) {
    if (lodash.isEqual(prevProps.graphData, this.props.graphData)) {
      if (prevProps.width === this.props.width &&
        prevProps.height === this.props.height &&
        lodash.isEqual(prevProps.logData, this.props.logData)) {
        // perhaps only display step is changed
        this.applyMutableWeight();
      } else {
        // something changed, re-render
        this.renderGraph(false);
      }
    } else {
      // re-calculate collapse
      this.renderGraph(true);
    }
  }

  private graphElements() {
    let graphElements = [];
    let nodeSet = new Set();
    let collapsedNodes = Object.keys(this.collapsedNodes).filter((d) => this.collapsedNodes[d]);
    const collapseMap = collapseTo(this.props.graphData, collapsedNodes);
    for (const node of this.props.graphData['node']) {
      nodeSet.add(node['name']);
    }
    for (const cluster of this.props.graphData['cluster']) {
      if (collapseMap[cluster['name']] !== cluster['name'])
        continue;
      const isCompound = collapsedNodes.indexOf(cluster['name']) === -1;
      let clusterData: any = {
        id: cluster['name'],
        label: cluster['tail']
      };
      if (cluster['parent']) {
        clusterData.parent = cluster['parent'];
      }
      graphElements.push({
        data: clusterData,
        classes: isCompound ? ['compound'] : []
      });
    }
    for (const node of this.props.graphData['node']) {
      if (collapseMap[node['name']] !== node['name'])
        continue;
      let nodeData: any = {
        id: node['name'],
        label: node['op']
      };
      if (node['parent']) {
        nodeData.parent = node['parent'];
      }
      graphElements.push({
        data: nodeData,
        classes: ['op']
      });
    }
    for (const node of this.props.graphData['node']) {
      if (!node.hasOwnProperty('input'))
        continue;
      const target = node['name'];
      for (const input of node['input']) {
        if (!nodeSet.has(input))
          continue;
        if (input !== target && collapseMap[input] === collapseMap[target])
          continue;
        graphElements.push({
          data: {
            id: JSON.stringify([input, target]),
            source: collapseMap[input],
            target: collapseMap[target]
          }
        });
      }
    }
    return graphElements;
  }

  private applyMutableWeight() {
    const keyMap = this.props.logData[this.props.displayStep];
    const graph = this.props.graphData;
    let elemWeight: any = Object.create({});
    console.log(graph.mutableEdges);
    Object.entries(keyMap).forEach((entry) => {
      const [key, weights] = entry;
      console.log(weights);
      graph.mutableEdges[key].forEach((edges: any, i: number) => {
        edges.forEach((edge: any) => {
          if (!elemWeight.hasOwnProperty(edge.id())) {
            elemWeight[edge.id()] = (weights as any)[i];
          } else {
            elemWeight[edge.id()] += (weights as any)[i];
          }
        })
      });
    });
    console.log(elemWeight);
    Object.keys(elemWeight).forEach((k) => {
      elemWeight[k] = Math.min(elemWeight[k], 1.);
    });
    this.cyInstance!.edges()
      .filter(edge => !elemWeight.hasOwnProperty(edge.id()))
      .forEach(edge => {
        elemWeight[edge.id()] = 1.;
      });
    this.cyInstance!.nodes().forEach(node => {
      if (node.isOrphan()) {
        elemWeight[node.id()] = 1.;
      } else {
        const nodeWeight = node.connectedEdges()
          .map(edge => elemWeight[edge.id()] as number)
          .reduce((a, b) => Math.max(a, b), 0.);
        elemWeight[node.id()] = nodeWeight;
      }
    });
    this.cyInstance!.nodes()
      .filter(node => node.isParent())
      .forEach(node => {
        const compoundWeight = node.descendants()
          .filter(node => node.isChildless())
          .map(node => elemWeight[node.id()] as number)
          .reduce((a, b) => Math.max(a, b));
        elemWeight[node.id()] = compoundWeight;
      });
    Object.entries(elemWeight).forEach((entry) => {
      const [elem, weight] = entry;
      if (!this.elemWeight.hasOwnProperty(elem) || this.elemWeight[elem] !== weight) {
        this.cyInstance!.getElementById(elem).style({
          opacity: 0.2 + 0.8 * (weight as number)
        });
      }
    });
    console.log(elemWeight);
    this.elemWeight = elemWeight;
  }

  private renderGraph(graphChanged: boolean) {
    if (this.props.graphData === null || this.props.logData.length === 0)
      return;

    if (graphChanged) {
      const collapsed = defaultCollapsedNodes(this.props.graphData);
      this.collapsedNodes = Object.create({});
      collapsed.forEach((name) => { this.collapsedNodes[name] = true; });
      console.log(this.collapsedNodes);
    }
    const graphEl = this.graphElements();
    console.log(graphEl);
    this.cyInstance!.json({
      elements: graphEl
    });
    const layout = {
      name: 'dagre',
      animate: true,
      animationDuration: 1000,
    };
    this.cyInstance!.layout(layout).run();
    this.elemWeight = Object.create({});
    this.cyInstance!.elements().forEach((ele) => {
      this.elemWeight[ele.id()] = 1.;
    });
    this.applyMutableWeight();
  }

  render() {
    return (
      <div className='container' ref={this.container}
        style={{
          left: 0,
          top: 0,
          position: 'absolute',
          width: this.props.width - 15,
          height: this.props.height,
          overflow: 'hidden'
        }}>
      </div>
    );
  }
}
