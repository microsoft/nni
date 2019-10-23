import * as React from 'react';
import { Row, Modal } from 'antd';
import ReactEcharts from 'echarts-for-react';
import IntermediateVal from '../public-child/IntermediateVal';
import { TRIALS } from '../../static/datamodel';
import '../../static/style/compare.scss';
import { TableRecord, Intermedia, TooltipForIntermediate } from 'src/static/interface';

// the modal of trial compare
interface CompareProps {
    compareRows: Array<TableRecord>;
    visible: boolean;
    cancelFunc: () => void;
}

class Compare extends React.Component<CompareProps, {}> {

    public _isCompareMount: boolean;
    constructor(props: CompareProps) {
        super(props);
    }

    intermediate = () => {
        const { compareRows } = this.props;
        const trialIntermediate: Array<Intermedia> = [];
        const idsList: Array<string> = [];
        Object.keys(compareRows).map(item => {
            const temp = compareRows[item];
            const trial = TRIALS.getTrial(temp.id);
            trialIntermediate.push({
                name: temp.id,
                data: trial.description.intermediate,
                type: 'line',
                hyperPara: trial.description.parameters
            });
            idsList.push(temp.id);
        });
        // find max intermediate number
        trialIntermediate.sort((a, b) => { return (b.data.length - a.data.length); });
        const legend: Array<string> = [];
        // max length
        const length = trialIntermediate[0] !== undefined ? trialIntermediate[0].data.length : 0;
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
            legend: {
                data: idsList
            },
            xAxis: {
                type: 'category',
                // name: '# Intermediate',
                boundaryGap: false,
                data: xAxis
            },
            yAxis: {
                type: 'value',
                name: 'Metric'
            },
            series: trialIntermediate
        };
        return (
            <ReactEcharts
                option={option}
                style={{ width: '100%', height: 418, margin: '0 auto' }}
                notMerge={true} // update now
            />
        );

    }

    // render table column ---
    initColumn = () => {
        const idList: Array<string> = [];
        const sequenceIdList: Array<number> = [];
        const durationList: Array<number> = [];

        const compareRows = this.props.compareRows.map(tableRecord => TRIALS.getTrial(tableRecord.id));

        const parameterList: Array<object> = [];
        let parameterKeys: Array<string> = [];
        if (compareRows.length !== 0) {
            parameterKeys = Object.keys(compareRows[0].description.parameters);
        }
        Object.keys(compareRows).map(item => {
            const temp = compareRows[item];
            idList.push(temp.id);
            sequenceIdList.push(temp.sequenceId);
            durationList.push(temp.duration);
            parameterList.push(temp.description.parameters);
        });
        let isComplexSearchSpace;
        if (parameterList.length > 0) {
            isComplexSearchSpace = (typeof parameterList[0][parameterKeys[0]] === 'object')
                ? true : false;
        }
        return (
            <table className="compare">
                <tbody>
                    <tr>
                        <td className="column">Id</td>
                        {Object.keys(idList).map(key => {
                            return (
                                <td className="value idList" key={key}>{idList[key]}</td>
                            );
                        })}
                    </tr>
                    <tr>
                        <td className="column">Trial No.</td>
                        {Object.keys(sequenceIdList).map(key => {
                            return (
                                <td className="value idList" key={key}>{sequenceIdList[key]}</td>
                            );
                        })}
                    </tr>
                    <tr>
                        <td className="column">Default metric</td>
                        {Object.keys(compareRows).map(index => {
                            const temp = compareRows[index];
                            return (
                                <td className="value" key={index}>
                                    <IntermediateVal trialId={temp.id} />
                                </td>
                            );
                        })}
                    </tr>
                    <tr>
                        <td className="column">duration</td>
                        {Object.keys(durationList).map(index => {
                            return (
                                <td className="value" key={index}>{durationList[index]}</td>
                            );
                        })}
                    </tr>
                    {
                        isComplexSearchSpace
                            ?
                            null
                            :
                            Object.keys(parameterKeys).map(index => {
                                return (
                                    <tr key={index}>
                                        <td className="column" key={index}>{parameterKeys[index]}</td>
                                        {
                                            Object.keys(parameterList).map(key => {
                                                return (
                                                    <td key={key} className="value">
                                                        {parameterList[key][parameterKeys[index]]}
                                                    </td>
                                                );
                                            })
                                        }
                                    </tr>
                                );
                            })
                    }
                </tbody>
            </table>
        );
    }

    componentDidMount() {
        this._isCompareMount = true;
    }

    componentWillUnmount() {
        this._isCompareMount = false;
    }

    render() {
        const { visible, cancelFunc } = this.props;

        return (
            <Modal
                title="Compare trials"
                visible={visible}
                onCancel={cancelFunc}
                footer={null}
                destroyOnClose={true}
                maskClosable={false}
                width="90%"
            >
                <Row className="compare-intermediate">
                    {this.intermediate()}
                    <Row className="compare-yAxis"># Intermediate result</Row>
                </Row>
                <Row>{this.initColumn()}</Row>
            </Modal>
        );
    }
}

export default Compare;
