import * as React from 'react';
import { Switch } from 'antd';
import ReactEcharts from 'echarts-for-react';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { Trial } from '../../static/model/trial';
import { TooltipForAccuracy } from '../../static/interface';
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
}

class DefaultPoint extends React.Component<DefaultPointProps, DefaultPointState> {
    constructor(props: DefaultPointProps) {
        super(props);
        this.state = { bestCurveEnabled: false };
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
        const graph = generateGraphConfig(trials[trials.length - 1].sequenceId);
        if (this.state.bestCurveEnabled) {
            (graph as any).series = [ generateBestCurveSeries(trials), generateScatterSeries(trials) ];
        } else {
            (graph as any).series = [ generateScatterSeries(trials) ];
        }
        return graph;
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

function generateGraphConfig(maxSequenceId: number) {
    return {
        grid: {
            left: '8%',
        },
        tooltip: {
            trigger: 'item',
            enterable: true,
            position: (point: Array<number>, data: TooltipForAccuracy) => (
                [ (data.data[0] < maxSequenceId ? point[0] : (point[0] - 300)), 80 ]
            ),
            formatter: (data: TooltipForAccuracy) => (
                '<div class="tooldetailAccuracy">' +
                '<div>Trial No.: ' + data.data[0] + '</div>' +
                '<div>Default metric: ' + data.data[1] + '</div>' +
                '<div>Parameters: <pre>' + JSON.stringify(data.data[2], null, 4) + '</pre></div>' +
                '</div>'
            ),
        },
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
    const data = [[ best.sequenceId, best.accuracy, best.info.hyperParameters ]];

    for (let i = 1; i < trials.length; i++) {
        const trial = trials[i];
        const delta = trial.accuracy! - best.accuracy!;
        const better = (EXPERIMENT.optimizeMode === 'minimize') ? (delta < 0) : (delta > 0);
        if (better) {
            data.push([ trial.sequenceId, trial.accuracy, trial.info.hyperParameters ]);
            best = trial;
        } else {
            data.push([ trial.sequenceId, best.accuracy, trial.info.hyperParameters ]);
        }
    }

    return {
        type: 'line',
        lineStyle: { color: '#FF6600' },
        data,
    };
}

export default DefaultPoint;
