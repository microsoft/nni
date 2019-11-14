import * as React from 'react';
import { Switch } from 'antd';
import ReactEcharts from 'echarts-for-react';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { Trial } from '../../static/model/trial';
import { TooltipForAccuracy, EventMap } from '../../static/interface';
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface DefaultPointProps {
    trialIds: string[];
    visible: boolean;
    trialsUpdateBroadcast: number;
}

interface DefaultPointState {
    bestCurveEnabled: boolean;
    start: number; // for record data zoom
    end: number;
    startY: number;  // dataZoomY
    endY: number;
}

class DefaultPoint extends React.Component<DefaultPointProps, DefaultPointState> {
    constructor(props: DefaultPointProps) {
        super(props);
        this.state = {
            bestCurveEnabled: false,
            start: 0, // dataZoomX
            end: 100,
            startY: 0, // dataZoomY
            endY: 100,
        };
    }

    loadDefault = (checked: boolean) => {
        this.setState({ bestCurveEnabled: checked });
    }

    shouldComponentUpdate(nextProps: DefaultPointProps, nextState: DefaultPointState) {
        return nextProps.visible;
    }

    render() {
        const graph = this.generateGraph();
        const accNodata = (graph === EmptyGraph ? 'No data' : '');
        const onEvents = { 'dataZoom': this.metricDataZoom };

        return (
            <div>
                <div className="default-metric">
                    <div className="position">
                        <span className="bold">Optimization curve</span>
                        <Switch defaultChecked={false} onChange={this.loadDefault} />
                    </div>
                </div>
                <ReactEcharts
                    option={graph}
                    style={{
                        width: '100%',
                        height: 402,
                        margin: '0 auto',
                    }}
                    theme="my_theme"
                    notMerge={true} // update now
                    onEvents={onEvents}
                />
                <div className="showMess">{accNodata}</div>
            </div>
        );
    }

    private generateGraph() {
        const trials = TRIALS.getTrials(this.props.trialIds).filter(trial => trial.sortable);
        if (trials.length === 0) {
            return EmptyGraph;
        }
        const graph = this.generateGraphConfig(trials[trials.length - 1].sequenceId);
        if (this.state.bestCurveEnabled) {
            (graph as any).series = [generateBestCurveSeries(trials), generateScatterSeries(trials)];
        } else {
            (graph as any).series = [generateScatterSeries(trials)];
        }
        return graph;
    }

    private generateGraphConfig(maxSequenceId: number) {
        const { start, end, startY, endY } = this.state;
        return {
            grid: {
                left: '8%',
            },
            tooltip: {
                trigger: 'item',
                enterable: true,
                position: (point: Array<number>, data: TooltipForAccuracy) => (
                    [(data.data[0] < maxSequenceId ? point[0] : (point[0] - 300)), 80]
                ),
                formatter: (data: TooltipForAccuracy) => (
                    '<div class="tooldetailAccuracy">' +
                    '<div>Trial No.: ' + data.data[0] + '</div>' +
                    '<div>Default metric: ' + data.data[1] + '</div>' +
                    '<div>Parameters: <pre>' + JSON.stringify(data.data[2], null, 4) + '</pre></div>' +
                    '</div>'
                ),
            },
            dataZoom: [
                {
                    id: 'dataZoomX',
                    type: 'inside',
                    xAxisIndex: [0],
                    filterMode: 'empty',
                    start: start,
                    end: end
                },
                {
                    id: 'dataZoomY',
                    type: 'inside',
                    yAxisIndex: [0],
                    filterMode: 'empty',
                    start: startY,
                    end: endY
                }
            ],
            xAxis: {
                name: 'Trial',
                type: 'category',
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true,
            },
            series: undefined,
        };
    }

    private metricDataZoom = (e: EventMap) => {
        if (e.batch !== undefined) {
            if (e.batch[0].dataZoomId !== undefined) {
                if (e.batch[0].dataZoomId === 'dataZoomX') {
                    this.setState(() => ({ start: e.batch[0].start, end: e.batch[0].end }));
                }
            }
            if (e.batch[1] !== undefined) {
                if (e.batch[1].dataZoomId !== undefined) {
                    if (e.batch[1].dataZoomId === 'dataZoomY') {
                        this.setState(() => ({ startY: e.batch[1].start, endY: e.batch[1].end }));
                    }
                }
            }
        }
    }
}

const EmptyGraph = {
    grid: {
        left: '8%'
    },
    xAxis: {
        name: 'Trial',
        type: 'category',
    },
    yAxis: {
        name: 'Default metric',
        type: 'value',
    }
};

function generateScatterSeries(trials: Trial[]) {
    const data = trials.map(trial => [
        trial.sequenceId,
        trial.accuracy,
        trial.description.parameters,
    ]);
    return {
        symbolSize: 6,
        type: 'scatter',
        data,
    };
}

function generateBestCurveSeries(trials: Trial[]) {
    let best = trials[0];
    const data = [[best.sequenceId, best.accuracy, best.description.parameters]];

    for (let i = 1; i < trials.length; i++) {
        const trial = trials[i];
        const delta = trial.accuracy! - best.accuracy!;
        const better = (EXPERIMENT.optimizeMode === 'minimize') ? (delta < 0) : (delta > 0);
        if (better) {
            data.push([trial.sequenceId, trial.accuracy, trial.description.parameters]);
            best = trial;
        } else {
            data.push([trial.sequenceId, best.accuracy, trial.description.parameters]);
        }
    }

    return {
        type: 'line',
        lineStyle: { color: '#FF6600' },
        data,
    };
}

export default DefaultPoint;
