import * as React from 'react';
import { Switch } from 'antd';
import ReactEcharts from 'echarts-for-react';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { Trial } from '../../static/model/trial';
import { TooltipForAccuracy, EventMap } from '../../static/interface';
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
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
        scale: true,
    }
};
interface DefaultPointProps {
    trialIds: string[];
    visible: boolean;
    trialsUpdateBroadcast: number;
}

interface DefaultPointState {
    bestCurveEnabled: boolean;
    startY: number;  // dataZoomY
    endY: number;
}

class DefaultPoint extends React.Component<DefaultPointProps, DefaultPointState> {
    constructor(props: DefaultPointProps) {
        super(props);
        this.state = {
            bestCurveEnabled: false,
            startY: 0, // dataZoomY
            endY: 100,
        };
    }

    loadDefault = (checked: boolean): void => {
        this.setState({ bestCurveEnabled: checked });
    }

    shouldComponentUpdate(nextProps: DefaultPointProps): boolean {
        return nextProps.visible;
    }

    generateScatterSeries = (trials: Trial[]): any => {
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
    
    generateBestCurveSeries = (trials: Trial[]): any => {
        let best = trials[0];
        const data = [[best.sequenceId, best.accuracy, best.description.parameters]];
    
        for (let i = 1; i < trials.length; i++) {
            const trial = trials[i];
            if (trial.accuracy !== undefined) {
                if (best.accuracy !== undefined) {
                    const delta = trial.accuracy - best.accuracy;
                    const better = (EXPERIMENT.optimizeMode === 'minimize') ? (delta < 0) : (delta > 0);
                    if (better) {
                        data.push([trial.sequenceId, trial.accuracy, trial.description.parameters]);
                        best = trial;
                    } else {
                        data.push([trial.sequenceId, best.accuracy, trial.description.parameters]);
                    }
                }
            }
        }
    
        return {
            type: 'line',
            lineStyle: { color: '#FF6600' },
            data,
        };
    }

    render(): React.ReactNode {
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

    private generateGraph(): any {
        const trials = TRIALS.getTrials(this.props.trialIds).filter(trial => trial.sortable);
        if (trials.length === 0) {
            return EmptyGraph;
        }
        const graph = this.generateGraphConfig(trials[trials.length - 1].sequenceId);
        if (this.state.bestCurveEnabled) {
            (graph as any).series = [this.generateBestCurveSeries(trials), this.generateScatterSeries(trials)];
        } else {
            (graph as any).series = [this.generateScatterSeries(trials)];
        }
        return graph;
    }

    private generateGraphConfig(maxSequenceId: number): any {
        const { startY, endY } = this.state;
        return {
            grid: {
                left: '8%',
            },
            tooltip: {
                trigger: 'item',
                enterable: true,
                position: (point: Array<number>, data: TooltipForAccuracy): Array<number> => (
                    [(data.data[0] < maxSequenceId ? point[0] : (point[0] - 300)), 80]
                ),
                formatter: (data: TooltipForAccuracy): any => (
                    '<div class="tooldetailAccuracy">' +
                    '<div>Trial No.: ' + data.data[0] + '</div>' +
                    '<div>Default metric: ' + data.data[1] + '</div>' +
                    '<div>Parameters: <pre>' + JSON.stringify(data.data[2], null, 4) + '</pre></div>' +
                    '</div>'
                ),
            },
            dataZoom: [
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

    private metricDataZoom = (e: EventMap): void => {
        if (e.batch !== undefined) {
            this.setState(() => ({
                startY: (e.batch[0].start !== null ? e.batch[0].start : 0),
                endY: (e.batch[0].end !== null ? e.batch[0].end : 100)
            }));
        }
    }
}

export default DefaultPoint;
