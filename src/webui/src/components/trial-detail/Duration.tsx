import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { TableObj } from 'src/static/interface';
import { filterDuration } from 'src/static/function';
require('echarts/lib/chart/bar');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface Runtrial {
    trialId: Array<string>;
    trialTime: Array<number>;
}

interface DurationProps {
    source: Array<TableObj>;
    whichGraph: string;
}

interface DurationState {
    durationSource: {};
}

class Duration extends React.Component<DurationProps, DurationState> {

    public _isMounted = false;

    constructor(props: DurationProps) {

        super(props);
        this.state = {
            durationSource: this.initDuration(this.props.source),
        };

    }

    initDuration = (source: Array<TableObj>) => {
        const trialId: Array<string> = [];
        const trialTime: Array<number> = [];
        const trialJobs = source.filter(filterDuration);
        Object.keys(trialJobs).map(item => {
            const temp = trialJobs[item];
            trialId.push(temp.sequenceId);
            trialTime.push(temp.duration);
        });
        return {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                bottom: '3%',
                containLabel: true,
                left: '1%',
                right: '4%'
            },

            dataZoom: [{
                type: 'slider',
                name: 'trial',
                filterMode: 'filter',
                yAxisIndex: 0,
                orient: 'vertical'
            }, {
                type: 'slider',
                name: 'trial',
                filterMode: 'filter',
                xAxisIndex: 0
            }],
            xAxis: {
                name: 'Time',
                type: 'value',
            },
            yAxis: {
                name: 'Trial',
                type: 'category',
                data: trialId
            },
            series: [{
                type: 'bar',
                data: trialTime
            }]
        };
    }

    getOption = (dataObj: Runtrial) => {
        return {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                bottom: '3%',
                containLabel: true,
                left: '1%',
                right: '4%'
            },

            dataZoom: [{
                type: 'slider',
                name: 'trial',
                filterMode: 'filter',
                yAxisIndex: 0,
                orient: 'vertical'
            }, {
                type: 'slider',
                name: 'trial',
                filterMode: 'filter',
                xAxisIndex: 0
            }],
            xAxis: {
                name: 'Time',
                type: 'value',
            },
            yAxis: {
                name: 'Trial',
                type: 'category',
                data: dataObj.trialId
            },
            series: [{
                type: 'bar',
                data: dataObj.trialTime
            }]
        };
    }

    drawDurationGraph = (source: Array<TableObj>) => {
        // why this function run two times when props changed?
        const trialId: Array<string> = [];
        const trialTime: Array<number> = [];
        const trialRun: Array<Runtrial> = [];
        const trialJobs = source.filter(filterDuration);
        Object.keys(trialJobs).map(item => {
            const temp = trialJobs[item];
            trialId.push(temp.sequenceId);
            trialTime.push(temp.duration);
        });
        trialRun.push({
            trialId: trialId,
            trialTime: trialTime
        });
        if (this._isMounted) {
            this.setState({
                durationSource: this.getOption(trialRun[0])
            });
        }
    }

    componentDidMount() {
        this._isMounted = true;
        const { source } = this.props;
        this.drawDurationGraph(source);
    }

    componentWillReceiveProps(nextProps: DurationProps) {
        const { whichGraph, source } = nextProps;
        if (whichGraph === '3') {
            this.drawDurationGraph(source);
        }
    }

    shouldComponentUpdate(nextProps: DurationProps, nextState: DurationState) {
    
        const { whichGraph, source } = nextProps;
        if (whichGraph === '3') { 
            const beforeSource = this.props.source;
            if (whichGraph !== this.props.whichGraph) {
                return true;
            }

            if (source.length !== beforeSource.length) {
                return true;
            }
            
            if (source[source.length - 1].duration !== beforeSource[beforeSource.length - 1].duration) {
                return true;
            }

            if (source[source.length - 1].status !== beforeSource[beforeSource.length - 1].status) {
                return true;
            }
        }
        return false;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { durationSource } = this.state;
        return (
            <div>
                <ReactEcharts
                    option={durationSource}
                    style={{ width: '95%', height: 412, margin: '0 auto' }}
                    theme="my_theme"
                    notMerge={true} // update now
                />
            </div>
        );
    }
}

export default Duration;