import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../../static/const';
import ReactEcharts from 'echarts-for-react';
const echarts = require('echarts/lib/echarts');
require('echarts/lib/chart/bar');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});

interface Runtrial {
    trialId: Array<string>;
    trialTime: Array<number>;
}

interface DurationState {
    durationSource: {};
}

class Duration extends React.Component<{}, DurationState> {

    static intervalDuration = 1;
    public _isMounted = false;

    constructor(props: {}) {
        super(props);

        this.state = {

            durationSource: {}
        };

    }

    getOption = (dataObj: Runtrial) => {
        let xAxis = dataObj.trialTime;
        let yAxis = dataObj.trialId;
        let option = {
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
                data: yAxis
            },
            series: [{
                type: 'bar',
                data: xAxis
            }]
        };
        return option;
    }

    drawRunGraph = () => {

        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const trialId: Array<string> = [];
                    const trialTime: Array<number> = [];
                    const trialRun: Array<Runtrial> = [];
                    Object.keys(trialJobs).map(item => {
                        if (trialJobs[item].status !== 'WAITING') {
                            let duration: number = 0;
                            const end = trialJobs[item].endTime;
                            const start = trialJobs[item].startTime;
                            if (start && end) {
                                duration = (end - start) / 1000;
                            } else {
                                duration = (new Date().getTime() - start) / 1000;
                            }
                            trialId.push(trialJobs[item].sequenceId);
                            trialTime.push(duration);
                        }
                    });
                    trialRun.push({
                        trialId: trialId,
                        trialTime: trialTime
                    });
                    if (this._isMounted && res.status === 200) {
                        this.setState({
                            durationSource: this.getOption(trialRun[0])
                        });
                    }
                }
            });
    }

    componentDidMount() {

        this._isMounted = true;
        this.drawRunGraph();
        Duration.intervalDuration = window.setInterval(this.drawRunGraph, 10000);
    }

    componentWillUnmount() {

        this._isMounted = false;
        window.clearInterval(Duration.intervalDuration);
    }

    render() {
        const { durationSource } = this.state;
        return (
            <div>
                <ReactEcharts
                    option={durationSource}
                    style={{ width: '100%', height: 412, margin: '0 auto' }}
                    theme="my_theme"
                />
            </div>
        );
    }
}

export default Duration;