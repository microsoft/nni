import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { TableObj } from 'src/static/interface';
require('echarts/lib/chart/bar');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface Runtrial {
    trialId: Array<string>;
    trialTime: Array<number>;
}

interface DurationProps {
    source: Array<TableObj>;
}

interface DurationState {
    durationSource: {};
}

class Duration extends React.Component<DurationProps, DurationState> {

    public _isMounted = false;

    constructor(props: DurationProps) {

        super(props);
        this.state = {
            durationSource: {}
        };

    }

    getOption = (dataObj: Runtrial) => {
        return  {
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

    drawDurationGraph = (trialJobs: Array<TableObj>) => {

        const trialId: Array<string> = [];
        const trialTime: Array<number> = [];
        const trialRun: Array<Runtrial> = [];
        Object.keys(trialJobs).map(item => {
            const temp = trialJobs[item];
            if (temp.status !== 'WAITING') {
                trialId.push(temp.sequenceId);
                trialTime.push(temp.duration);
            }
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

    componentWillReceiveProps(nextProps: DurationProps) {
        const trialJobs = nextProps.source;
        this.drawDurationGraph(trialJobs);
    }

    componentDidMount() {
        this._isMounted = true;
        // init: user don't search
        const {source} = this.props;
        this.drawDurationGraph(source);
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
                />
            </div>
        );
    }
}

export default Duration;