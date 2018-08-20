import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../const';
import ReactEcharts from 'echarts-for-react';
const echarts = require('echarts/lib/echarts');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('../style/accuracy.css');

const accStyle = {
    width: '100%',
    height: 600,
    margin: '0 auto'
};

interface ChartState {
    option: object;
    accNodata: string;
}

interface AccurPoint {
    yAxis: Array<number>;
}

class Accuracy extends React.Component<{}, ChartState> {

    public _isMounted = false;
    public intervalID = 0;

    constructor(props: {}) {

        super(props);
        this.state = {
            option: {},
            accNodata: ''
        };
    }

    getOption = (dataObj: AccurPoint) => {
        const yAxis = dataObj.yAxis;
        const xAxis: Array<number> = [];
        for (let i = 1; i <= yAxis.length; i++) {
            xAxis.push(i);
        }
        return {
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
                data: xAxis
            },
            yAxis: {
                name: 'Accuracy',
                type: 'value',
                min: 0,
                max: 1,
                data: yAxis
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: yAxis
            }]
        };
    }

    drawPointGraph = () => {

        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    const accData = res.data;
                    const accArr: Array<number> = [];
                    const accY: Array<AccurPoint> = [];
                    Object.keys(accData).map(item => {
                        if (accData[item].status === 'SUCCEEDED' && accData[item].finalMetricData) {
                            accArr.push(parseFloat(accData[item].finalMetricData.data));
                        }
                    });
                    accY.push({yAxis: accArr});
                    let optionObj = this.getOption(accY[0]);
                    this.setState({ option: optionObj }, () => {
                        if (accArr.length === 0) {
                            this.setState({
                                accNodata: 'No data'
                            });
                        } else {
                            this.setState({
                                accNodata: ''
                            });
                        }
                    });
                }
            });
    }

    componentDidMount() {

        this._isMounted = true;
        this.drawPointGraph();
        this.intervalID = window.setInterval(this.drawPointGraph, 10000);
    }
    componentWillUnmount() {

        this._isMounted = false;
        window.clearInterval(this.intervalID);
    }
    render() {
        const { accNodata, option } = this.state;
        return (
            <div className="graph">
                <div className="trial">
                    <div className="title">
                        <div>Trial Accuracy</div>
                    </div>
                    <div>
                        <ReactEcharts
                            option={option}
                            style={accStyle}
                            theme="my_theme"
                        />
                        <div className="showMess">{accNodata}</div>
                    </div>
                </div>
            </div>
        );
    }
}

export default Accuracy;