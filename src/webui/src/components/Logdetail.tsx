import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../const';
import {
    message,
    Tabs,
    Button
} from 'antd';
const TabPane = Tabs.TabPane;
import '../style/logdetail.css';

interface LogState {
    trialId: string;
    slotLog: string;
    processLog: string;
}

class Logdetail extends React.Component<{}, LogState> {

    public _isMounted = false;

    constructor(props: {}) {

        super(props);
        this.state = {
            trialId: '',
            slotLog: '',
            processLog: ''
        };
    }

    getJobLog = () => {

        Object.keys(this.props).map(item => {
            if (item === 'location') {
                if (this._isMounted) {
                    this.setState({ trialId: this.props[item].state }, () => {

                        const { trialId } = this.state;
                        let id = trialId;

                        axios(`${MANAGER_IP}/jobLog`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json;charset=utf-8'
                            },
                            data: {
                                id
                            }
                        })
                            .then(res => {
                                if (res.status === 200 && this._isMounted) {
                                    this.setState({
                                        slotLog: res.data.trial_slot_log,
                                        processLog: res.data.trial_process_log
                                    });
                                }
                            });
                    });
                }
            }
        });
    }

    getPaiDetail = (id: string) => {

        axios(`${MANAGER_IP}/jobPaiPage`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json;charset=utf-8'
            },
            data: {
                id
            }
        })
            .then(res => {
                if (res.status === 200) {
                    message.success('Successful send');
                    setTimeout(this.openPage(res.data.url), 100);
                }
            });
    }

    openPage = (pailog: string) => {
        window.open(pailog);
    }

    paiLog = () => {

        axios(`${MANAGER_IP}/paiPage`, {
            method: 'POST'
        })
            .then(res => {
                if (res.status === 200) {
                    setTimeout(this.openPage(res.data.url), 200);
                }
            });
    }

    componentDidMount() {

        this._isMounted = true;
        this.getJobLog();
    }

    componentWillUnmount() {

        this._isMounted = false;
    }

    render() {
        const { trialId, slotLog, processLog } = this.state;
        return (
            <div className="log">
                <div>
                    <Tabs type="card">
                        <TabPane tab="trial_slot_log" key="1">
                            <pre>{slotLog}</pre>
                        </TabPane>
                        <TabPane tab="trial_process_log" key="2">
                            <pre>{processLog}</pre>
                        </TabPane>
                    </Tabs>
                </div>
                <div className="pai">
                    <Button
                        type="primary"
                        className="tableButton marginTab"
                        onClick={this.getPaiDetail.bind(this, trialId)}
                    >
                        pai
                    </Button>
                    <Button
                        type="primary"
                        className="tableButton"
                        onClick={this.paiLog}
                    >
                        main job log
                    </Button>
                </div>
            </div>
        );
    }
}

export default Logdetail;