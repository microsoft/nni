import React, { createRef } from 'react';
import './App.css';
import * as d3 from 'd3';
import 'typeface-roboto';
import dagreD3 from 'dagre-d3';
import lodash from 'lodash';

type ChartProps = {
  width: number,
  height: number,
  displayStep: number,
  graphData: any,
  logData: any[],
}

export default class Chart extends React.Component<ChartProps> {
  svg = createRef<SVGSVGElement>();
  innerG = createRef<SVGSVGElement>();
  zoom: any;

  componentDidMount() {
    const svg = d3.select(this.svg.current);
    const inner: any = d3.select(this.innerG.current);
    this.zoom = d3.zoom().on('zoom', function () {
      inner.attr('transform', d3.event.transform);
    });
    svg.call(this.zoom);
    this.renderGraph(true);
  }

  componentDidUpdate(prevProps: ChartProps) {
    const refresh = !lodash.isEqual(prevProps.graphData, this.props.graphData);
    this.renderGraph(refresh);
  }

  private getActiveChains() {
    const getEdgeKey = (edge: string[]): string => `${edge[0]}$${edge[1]}`;
    const currentSelection: any = this.props.logData[this.props.displayStep];
    let weightMap = new Map<string, number>();
    const key2chain: any = this.props.graphData['key2chain'];
    for (const entry of Object.entries(key2chain)) {
      // entry[0]: mutable key
      // entry[1]: list of list of edges str that will be affected by the selection
      (entry[1] as string[][][]).forEach((edge_list: string[][], i: number) => {
        const weight = currentSelection[entry[0]][i];
        for (const edge of edge_list) {
          const edgeKey = getEdgeKey(edge);
          if (weightMap.has(edgeKey)) {
            weightMap.set(edgeKey, weightMap.get(edgeKey) + weight);  // ok
          } else {
            weightMap.set(edgeKey, weight);
          }
        }
      });
    }
    for (const edge of this.props.graphData['edges']) {
      if (!weightMap.has(getEdgeKey(edge)))
        weightMap.set(getEdgeKey(edge), 1.);
    }
    for (const node of this.props.graphData['nodes'])
      weightMap.set(node['id'], -1.);
    for (const edge of this.props.graphData['edges']) {
      const edgeKey = getEdgeKey(edge);
      if (!weightMap.has(edgeKey))
        weightMap.set(edgeKey, 1.);
      const edgeWeight = weightMap.get(edgeKey);
      for (const node of edge) {
        const nodeWeight = Math.max(weightMap.get(node)!, edgeWeight!);
        weightMap.set(node, nodeWeight);
      }
    }
    for (const node of this.props.graphData['nodes'])
      if (weightMap.get(node['id'])! < 0)
        weightMap.set(node['id'], 1.);
    return weightMap;
  }

  private renderGraph(reset: boolean): dagreD3.graphlib.Graph | null {
    if (this.props.graphData === null || this.props.logData.length === 0)
      return null;
    const activations = this.getActiveChains();
    const svg = d3.select(this.svg.current);
    const inner: any = d3.select(this.innerG.current);
    const render = new dagreD3.render();
    let graph = new dagreD3.graphlib.Graph()
      .setGraph({})
      .setDefaultEdgeLabel(function () { return {}; });
    for (const node of this.props.graphData['nodes']) {
      const activation = activations.get(node['id'])!;
      graph.setNode(node['id'], {
        label: node['text'],
        class: node['type'],
        rx: 5,
        ry: 5,
        style: `opacity: ${Math.min(activation * 0.7 + 0.3, 1.)}`
      });
    }
    for (const edge of this.props.graphData['edges']) {
      const activation = activations.get(`${edge[0]}$${edge[1]}`)!;
      graph.setEdge(edge[0], edge[1], {
        style: `opacity: ${Math.min(activation * 0.7 + 0.3, 1.)}`
      });
    }
    render(inner, graph);

    if (reset) {
      const initialScale = 1.;
      const xCenterOffset = (+svg.attr('width') - graph.graph().width * initialScale) / 2;
      svg.call(this.zoom.transform, d3.zoomIdentity.translate(xCenterOffset, 20).scale(initialScale));
      svg.attr('height', graph.graph().height * initialScale + 40);
    }
    return graph;
  }

  render() {
    return (
      <svg className='container' ref={this.svg}
        width={this.props.width - 15} height={this.props.height}>
        <g ref={this.innerG} />
      </svg>
    );
  }
}
