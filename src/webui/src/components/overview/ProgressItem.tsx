import * as React from 'react';
import { Row, Col, Progress } from 'antd';

interface ProItemProps {
    who: string;
    percent: number;
    description: string;
    maxString: string;
    bgclass: string;
}

class ProgressBar extends React.Component<ProItemProps, {}> {

    constructor(props: ProItemProps) {
        super(props);

    }

    render() {
        const { who, percent, description, maxString, bgclass } = this.props;

        return (
            <div>
                <Row className={`probar ${bgclass}`}>
                        <Col span={8}>
                            <div className="name">{who}</div>
                        </Col>
                        <Col span={16} className="bar">
                            <div className="showProgress">
                                <Progress
                                    percent={percent}
                                    strokeWidth={30}
                                    // strokeLinecap={'square'}
                                    format={() => description}
                                />
                            </div>
                            <Row className="description">
                                <Col span={9}>0</Col>
                                <Col className="right" span={15}>{maxString}</Col>
                            </Row>
                        </Col>
                    </Row>
                    <br/>
            </div>
        );
    }
}

export default ProgressBar;