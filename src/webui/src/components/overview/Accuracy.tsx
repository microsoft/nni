import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
const echarts = require('echarts/lib/echarts');
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface AccuracyProps {
    accuracyData: object;
    accNodata: string;
    height: number;
}

class Accuracy extends React.Component<AccuracyProps, {}> {

    constructor(props: AccuracyProps) {
        super(props);

    }

    render() {
        const { accNodata, accuracyData, height } = this.props;
        return (
            <div>
                <ReactEcharts
                    option={accuracyData}
                    style={{
                        width: '90%',
                        height: height,
                        margin: '0 auto',
                    }}
                    theme="my_theme"
                />
                <div className="showMess">{accNodata}</div>
            </div>
        );
    }
}

export default Accuracy;