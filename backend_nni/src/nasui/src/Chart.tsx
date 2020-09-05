import React, { createRef } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import lodash from 'lodash';
import { Graph, NodeTs } from './graphUtils';

cytoscape.use(dagre);

type ChartProps = {
  width: number,
  height: number,
  graph: Graph | undefined,
  activation: any,
  handleSelectionChange: (_: string) => void,
  onRefresh: () => void,
  onRefreshComplete: () => void,
  layout: boolean,
  onLayoutComplete: () => void,
}

const styles = [
  {
    selector: ':active',
    style: {
      'overlay-opacity': 0.1,
    }
  },
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
      'background-opacity': 0.02,
      'border-color': '#555',
      'border-width': '2px',
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
      'border-color': '#101010',
    }
  },
  {
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'line-color': '#555',
      'target-arrow-color': '#555'
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
  expandSet: Set<string> = new Set<string>();
  elemWeight: Map<string, number> = new Map<string, number>();
  graphEl: any[] = [];
  private firstUpdate = true;

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
      const nodeId = e.target.id();
      const node = this.props.graph!.getNodeById(nodeId);
      if (node !== undefined && node.isParent()) {
        if (this.expandSet.has(nodeId)) {
          this.expandSet.delete(nodeId);
        } else {
          this.expandSet.add(nodeId);
        }
        this.renderGraph(false);
      }
    });
    this.cyInstance!.on('select', (e: cytoscape.EventObject) => {
      this.props.handleSelectionChange(e.target.id());
    });
    this.renderGraph(true);
  }

  componentDidUpdate(prevProps: ChartProps) {
    if (prevProps.graph === this.props.graph) {
      if (prevProps.width === this.props.width &&
        prevProps.height === this.props.height) {
        if (this.props.layout) {
          this.props.onLayoutComplete();
          this.reLayout();
        }
        if (prevProps.activation !== this.props.activation) {
          // perhaps only display step is changed
          this.applyMutableWeight();
        }
      } else {
        // something changed, re-render
        this.renderGraph(false);
      }
    } else {
      // re-calculate collapse
      this.renderGraph(true, this.firstUpdate);
      this.firstUpdate = false;
    }
  }

  private graphElements() {
    let graphElements: any[] = [];
    const { graph } = this.props;
    if (graph === undefined)
      return [];

    const collapseMap = new Map<string, string>();
    const traverse = (node: NodeTs, top: NodeTs | undefined) => {
      collapseMap.set(node.id, top === undefined ? node.id : top.id);
      if (node.id && (top === undefined || node === top)) {
        // not root and will display
        const isCompound = node.isParent() && this.expandSet.has(node.id);
        let data: any = {
          id: node.id,
          label: node.op ? node.op : node.tail,
        };
        if (node.parent !== undefined)
          data.parent = node.parent.id;
        const classes = [];
        if (isCompound) classes.push('compound');
        if (node.op) classes.push('op');
        graphElements.push({
          data: data,
          classes: classes
        });
      }
      for (const child of node.children)
        traverse(child, node.id && top === undefined && !this.expandSet.has(node.id) ? node : top);
    }
    traverse(graph.root, undefined);
    graph.edges.forEach(edge => {
      const [srcCollapse, trgCollapse] = [edge.source, edge.target].map((node) => collapseMap.get(node.id)!);
      if (edge.source !== edge.target && srcCollapse === trgCollapse) {
        return;
      }
      graphElements.push({
        data: {
          id: edge.id,
          source: srcCollapse,
          target: trgCollapse
        }
      });
    });
    return graphElements;
  }

  private applyMutableWeight() {
    const { graph, activation } = this.props;
    if (graph === undefined || activation === undefined)
      return;
    const weights = graph.weightFromMutables(activation);
    weights.forEach((weight, elem) => {
      if (this.elemWeight.get(elem) !== weight) {
        this.cyInstance!.getElementById(elem).style({
          opacity: 0.2 + 0.8 * weight
        });
      }
    });
    this.elemWeight = weights;
  }

  private graphElDifference(prev: any[], next: any[]): [Set<string>, any] {
    const tracedElements = new Set(prev.map(ele => ele.data.id));
    const prevMap = new Map(prev.map(ele => [ele.data.id, ele]));
    const nextMap = new Map(next.map(ele => [ele.data.id, ele]));
    const addedEles: any = [];
    nextMap.forEach((val, k) => {
      const prevEle = prevMap.get(k);
      if (prevEle === undefined) {
        addedEles.push(val);
      } else if (!lodash.isEqual(val, prevEle)) {
        tracedElements.delete(k);
        addedEles.push(val);
      } else {
        tracedElements.delete(k);
      }
    });
    return [tracedElements, addedEles];
  }

  private reLayout() {
    this.props.onRefresh();
    const _render = () => {
      const layout: any = {
        name: 'dagre'
      };
      this.cyInstance!.layout(layout).run();
      this.props.onRefreshComplete();
    };
    setTimeout(_render, 100);
  }

  private renderGraph(graphChanged: boolean, fit: boolean = false) {
    const { graph } = this.props;
    if (graph === undefined)
      return;

    this.props.onRefresh();
    const _render = () => {
      if (graphChanged)
        this.expandSet = lodash.cloneDeep(graph.defaultExpandSet);
      const graphEl = this.graphElements();
      const [remove, add] = this.graphElDifference(this.graphEl, graphEl);
      const layout: any = {
        name: 'dagre'
      };
      if (graphEl.length > 100) {
        if (remove.size > 0) {
          const removedEles = this.cyInstance!.elements().filter(ele => remove.has(ele.id()));
          this.cyInstance!.remove(removedEles);
        }
        if (add.length > 0) {
          const eles = this.cyInstance!.add(add);
          this.cyInstance!.json({
            elements: graphEl
          });
          if (!fit) {
            layout.fit = false;
          }
          eles.layout(layout).run();
        }
      } else {
        this.cyInstance!.json({
          elements: graphEl
        });
        this.cyInstance!.layout(layout).run();
      }
      this.applyMutableWeight();
      this.graphEl = graphEl;
      this.props.onRefreshComplete();
    };
    if (graph.nodes.length > 100)
      setTimeout(_render, 100);
    else
      _render();
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
