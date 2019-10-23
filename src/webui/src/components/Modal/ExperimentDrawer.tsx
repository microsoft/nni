import * as React from 'react';
import axios from 'axios';
import { downFile } from '../../static/function';
import { Drawer, Tabs, Row, Col, Button } from 'antd';
import { MANAGER_IP, DRAWEROPTION } from '../../static/const';
import MonacoEditor from 'react-monaco-editor';
const { TabPane } = Tabs;
import '../../static/style/logDrawer.scss';

interface ExpDrawerProps {
    isVisble: boolean;
    closeExpDrawer: () => void;
}

interface ExpDrawerState {
    experiment: string;
    expDrawerHeight: number;
}

class ExperimentDrawer extends React.Component<ExpDrawerProps, ExpDrawerState> {

    public _isCompareMount: boolean;
    constructor(props: ExpDrawerProps) {
        super(props);

        this.state = {
            experiment: '',
            expDrawerHeight: window.innerHeight - 48
        };
    }

    getExperimentContent = () => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/experiment`),
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/metric-data`)
            ])
            .then(axios.spread((res, res1, res2) => {
                if (res.status === 200 && res1.status === 200 && res2.status === 200) {
                    if (res.data.params.searchSpace) {
                        res.data.params.searchSpace = JSON.parse(res.data.params.searchSpace);
                    }
                    let trialMessagesArr = res1.data;
                    const interResultList = res2.data;
                    Object.keys(trialMessagesArr).map(item => {
                        // not deal with trial's hyperParameters
                        const trialId = trialMessagesArr[item].id;
                        // add intermediate result message
                        trialMessagesArr[item].intermediate = [];
                        Object.keys(interResultList).map(key => {
                            const interId = interResultList[key].trialJobId;
                            if (trialId === interId) {
                                trialMessagesArr[item].intermediate.push(interResultList[key]);
                            }
                        });
                    });
                    const result = {
                        experimentParameters: res.data,
                        trialMessage: trialMessagesArr
                    };
                    if (this._isCompareMount === true) {
                        this.setState({ experiment: JSON.stringify(result, null, 4) });
                    }
                }
            }));
    }

    downExperimentParameters = () => {
        const { experiment } = this.state;
        downFile(experiment, 'experiment.json');
    }

    onWindowResize = () => {
        this.setState(() => ({expDrawerHeight: window.innerHeight - 48}));
    }

    componentDidMount() {
        this._isCompareMount = true;
        this.getExperimentContent();
        window.addEventListener('resize', this.onWindowResize);
    }

    componentWillReceiveProps(nextProps: ExpDrawerProps) {
        const { isVisble } = nextProps;
        if (isVisble === true) {
            this.getExperimentContent();
        }
    }

    componentWillUnmount() {
        this._isCompareMount = false;
        window.removeEventListener('resize', this.onWindowResize);
    }

    render() {
        const { isVisble, closeExpDrawer } = this.props;
        const { experiment, expDrawerHeight } = this.state;
        return (
            <Row className="logDrawer">
                <Drawer
                    // title="Log Message"
                    placement="right"
                    closable={false}
                    destroyOnClose={true}
                    onClose={closeExpDrawer}
                    visible={isVisble}
                    width="54%"
                    height={expDrawerHeight}
                >
                    {/* 104: tabHeight(40) + tabMarginBottom(16) + buttonHeight(32) + buttonMarginTop(16) */}
                    <div className="card-container log-tab-body">
                        <Tabs type="card" style={{ height: expDrawerHeight, minHeight: 190 }}>
                            <TabPane tab="Experiment Parameters" key="Experiment">
                                <div className="just-for-log">
                                    <MonacoEditor
                                        width="100%"
                                        height={expDrawerHeight - 104}
                                        language="json"
                                        value={experiment}
                                        options={DRAWEROPTION}
                                    />
                                </div>
                                <Row className="buttons">
                                    <Col span={12}  className="download">
                                        <Button
                                            type="primary"
                                            onClick={this.downExperimentParameters}
                                        >
                                            Download
                                        </Button>
                                    </Col>
                                    <Col span={12} className="close">
                                        <Button
                                            type="default"
                                            onClick={closeExpDrawer}
                                        >
                                            Close
                                        </Button>
                                    </Col>
                                </Row>
                            </TabPane>
                        </Tabs>
                    </div>
                </Drawer>
            </Row>
        );
    }
}

export default ExperimentDrawer;
