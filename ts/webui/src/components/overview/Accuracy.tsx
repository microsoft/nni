import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import echarts from 'echarts/lib/echarts';
echarts.registerTheme('my_theme', {
    color: '#3c8dbc'
});
import 'echarts/lib/chart/scatter';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

interface AccuracyProps {
    accuracyData: object;
    accNodata: string;
}

class Accuracy extends React.Component<AccuracyProps, {}> {
    constructor(props: AccuracyProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { accNodata, accuracyData } = this.props;
        return (
            <div className='defaultMetricContainer'>
                <ReactEcharts option={accuracyData} theme='my_theme' />
                <div className='showMess'>{accNodata}</div>
            </div>
        );
    }
}

export default Accuracy;
