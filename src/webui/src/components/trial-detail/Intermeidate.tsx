import * as React from 'react';
import { Row, Col, Button, Switch } from 'antd';
import { TooltipForIntermediate, TableObj } from '../../static/interface';
import ReactEcharts from 'echarts-for-react';
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface Intermedia {
    name: string; // id
    type: string;
    data: Array<number | object>; // intermediate data
    hyperPara: object; // each trial hyperpara value
}
interface IntermediateState {
    interSource: object;
    filterSource: Array<TableObj>;
    eachIntermediateNum: number; // trial's intermediate number count
    isLoadconfirmBtn: boolean;
    isFilter: boolean;
}

interface IntermediateProps {
    source: Array<TableObj>;
}

class Intermediate extends React.Component<IntermediateProps, IntermediateState> {

    static intervalMediate = 1;
    public _isMounted = false;
    public pointInput: HTMLInputElement | null;
    public minValInput: HTMLInputElement | null;
    public maxValInput: HTMLInputElement | null;

    constructor(props: IntermediateProps) {
        super(props);
        this.state = {
            interSource: {},
            filterSource: [],
            eachIntermediateNum: 1,
            isLoadconfirmBtn: false,
            isFilter: false
        };
    }

    initMediate = () => {
        const option = {
            grid: {
                left: '5%',
                top: 40,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
            },
            yAxis: {
                type: 'value',
                name: 'Scape'
            }
        };
        if (this._isMounted) {
            this.setState(() => ({
                interSource: option
            }));
        }
    }

    drawIntermediate = (source: Array<TableObj>) => {
        if (source.length > 0) {
            const trialIntermediate: Array<Intermedia> = [];
            Object.keys(source).map(item => {
                const temp = source[item];
                trialIntermediate.push({
                    name: temp.id,
                    data: temp.description.intermediate,
                    type: 'line',
                    hyperPara: temp.description.parameters
                });
            });
            // find max intermediate number
            trialIntermediate.sort((a, b) => { return (b.data.length - a.data.length); });
            const legend: Array<string> = [];
            // max length
            const length = trialIntermediate[0].data.length;
            const xAxis: Array<number> = [];
            Object.keys(trialIntermediate).map(item => {
                const temp = trialIntermediate[item];
                legend.push(temp.name);
            });
            for (let i = 1; i <= length; i++) {
                xAxis.push(i);
            }
            const option = {
                tooltip: {
                    trigger: 'item',
                    enterable: true,
                    position: function (point: Array<number>, data: TooltipForIntermediate) {
                        if (data.dataIndex < length / 2) {
                            return [point[0], 80];
                        } else {
                            return [point[0] - 300, 80];
                        }
                    },
                    formatter: function (data: TooltipForIntermediate) {
                        const trialId = data.seriesName;
                        let obj = {};
                        const temp = trialIntermediate.find(key => key.name === trialId);
                        if (temp !== undefined) {
                            obj = temp.hyperPara;
                        }
                        return '<div class="tooldetailAccuracy">' +
                            '<div>Trial ID: ' + trialId + '</div>' +
                            '<div>Intermediate: ' + data.data + '</div>' +
                            '<div>Parameters: ' +
                            '<pre>' + JSON.stringify(obj, null, 4) + '</pre>' +
                            '</div>' +
                            '</div>';
                    }
                },
                grid: {
                    left: '5%',
                    top: 40,
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    name: 'Scape',
                    boundaryGap: false,
                    data: xAxis
                },
                yAxis: {
                    type: 'value',
                    name: 'Intermediate'
                },
                series: trialIntermediate
            };
            if (this._isMounted) {
                this.setState(() => ({
                    interSource: option
                }));
            }
        } else {
            this.initMediate();
        }
    }

    // confirm btn function [filter data]
    filterLines = () => {
        if (this._isMounted) {
            const filterSource: Array<TableObj> = [];
            this.setState({ isLoadconfirmBtn: true }, () => {
                const { source } = this.props;
                // get input value
                const pointVal = this.pointInput !== null ? this.pointInput.value : '';
                const minVal = this.minValInput !== null ? this.minValInput.value : '';
                const maxVal = this.maxValInput !== null ? this.maxValInput.value : '';
                // user not input message
                if (pointVal === '' || minVal === '') {
                    alert('Please input filter message');
                } else {
                    // user not input max value
                    const position = JSON.parse(pointVal);
                    const min = JSON.parse(minVal);
                    if (maxVal === '') {
                        Object.keys(source).map(item => {
                            const temp = source[item];
                            const val = temp.description.intermediate[position - 1];
                            if (val >= min) {
                                filterSource.push(temp);
                            }
                        });
                    } else {
                        const max = JSON.parse(maxVal);
                        Object.keys(source).map(item => {
                            const temp = source[item];
                            const val = temp.description.intermediate[position - 1];
                            if (val >= min && val <= max) {
                                filterSource.push(temp);
                            }
                        });
                    }
                    if (this._isMounted) {
                        this.setState({ filterSource: filterSource });
                    }
                    this.drawIntermediate(filterSource);
                }
                this.setState({ isLoadconfirmBtn: false });
            });
        }
    }

    switchTurn = (checked: boolean) => {
        if (this._isMounted) {
            this.setState({ isFilter: checked });
        }
        if (checked === false) {
            this.drawIntermediate(this.props.source);
        }
    }

    componentDidMount() {
        this._isMounted = true;
        const { source } = this.props;
        this.drawIntermediate(source);
    }

    componentWillReceiveProps(nextProps: IntermediateProps) {
        const { isFilter, filterSource } = this.state;
        if (isFilter === true) {
            const pointVal = this.pointInput !== null ? this.pointInput.value : '';
            const minVal = this.minValInput !== null ? this.minValInput.value : '';
            if (pointVal === '' && minVal === '') {
                this.drawIntermediate(nextProps.source);
            } else {
                this.drawIntermediate(filterSource);
            }
        } else {
            this.drawIntermediate(nextProps.source);
        }
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { interSource, isLoadconfirmBtn, isFilter } = this.state;

        return (
            <div>
                {/* style in para.scss */}
                <Row className="meline intermediate">
                    <Col span={8} />
                    <Col span={3} style={{ height: 34 }}>
                        {/* filter message */}
                        <span>filter</span>
                        <Switch
                            defaultChecked={false}
                            onChange={this.switchTurn}
                        />
                    </Col>
                    {
                        isFilter
                            ?
                            <div>
                                <Col span={3}>
                                    <span>Scape</span>
                                    <input
                                        placeholder="point"
                                        ref={input => this.pointInput = input}
                                        className="strange"
                                    />
                                </Col>
                                <Col className="range" span={10}>
                                    <span>Intermediate result</span>
                                    <input
                                        placeholder="number"
                                        ref={input => this.minValInput = input}
                                    />
                                    <span className="heng">-</span>
                                    <input
                                        placeholder="number"
                                        ref={input => this.maxValInput = input}
                                    />
                                    <Button
                                        type="primary"
                                        className="changeBtu tableButton"
                                        onClick={this.filterLines}
                                        disabled={isLoadconfirmBtn}
                                    >
                                        Confirm
                                    </Button>
                                </Col>
                            </div>
                            :
                            <Col />
                    }
                </Row>
                <Row>
                    <ReactEcharts
                        option={interSource}
                        style={{ width: '100%', height: 418, margin: '0 auto' }}
                        notMerge={true} // update now
                    />
                </Row>
            </div>
        );
    }
}

export default Intermediate;