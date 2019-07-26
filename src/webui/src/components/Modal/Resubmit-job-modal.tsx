import * as React from 'react';
import { Modal, Row, Button, Icon } from 'antd';
import './resubmit.scss';

// the modal of trial resubmit
interface ResubmitProps {
    isResubmitConfirm: boolean;
    isSucceedResubmit: boolean;
    isShowOk: boolean;
    reSubmitJob: () => void;
    cancelResubmit: () => void;
    changeSucceedState: (bool: boolean) => void;
}

class Resubmit extends React.Component<ResubmitProps, {}> {

    constructor(props: ResubmitProps) {
        super(props);
    }

    closeSucceedHint = () => {
        this.props.changeSucceedState(false);
    }

    render() {
        const { isResubmitConfirm, reSubmitJob, cancelResubmit, isShowOk, isSucceedResubmit } = this.props;
        return (
            <Row>
                <Modal
                    visible={isResubmitConfirm}
                    footer={null}
                    destroyOnClose={true}
                    maskClosable={false}
                    closable={false}
                    width="40%"
                    centered={true}
                >
                    <Row className="resubmit">
                        <h2 className="title"><Icon type="info-circle" />Resubmit trial</h2>
                        <div className="hint">
                            Are you sure you want to resubmit the trial?
                            If confirmed, We will apply for a new trial ID for you.
                        </div>
                        <Row className="buttons">
                            {/* confirm to resubmit job */}
                            <Button
                                type="primary"
                                className="tableButton padding-all"
                                onClick={reSubmitJob}
                            >
                                Confirm
                            </Button>
                            {/* cancel this choose */}
                            <Button
                                type="primary"
                                className="tableButton grey-bgcolor padding-all margin-cancel"
                                onClick={cancelResubmit}
                            >
                                Cancel
                            </Button>
                        </Row>
                    </Row>
                </Modal>

                <Modal
                    visible={isShowOk}
                    footer={null}
                    destroyOnClose={true}
                    maskClosable={false}
                    closable={false}
                    width="40%"
                    centered={true}
                >
                    <Row className="resubmit">
                        {
                            isSucceedResubmit
                                ?
                                <Row>
                                    <h2 className="title">
                                        <span>
                                            <Icon type="check-circle" className="color-succ" />Resubmit successfully
                                        </span>
                                    </h2>
                                    <div className="hint">
                                        <span>We have created a new trial.</span>
                                    </div>
                                </Row>
                                :
                                <Row>
                                    <h2 className="title">
                                        <span><Icon type="close-circle" className="color-error" />Resubmit Failed</span>
                                    </h2>
                                    <div className="hint">
                                        <span>500 error, fail to resubmit the job</span>
                                    </div>
                                </Row>
                        }

                        <Row className="buttons">
                            {/* close the modal */}
                            <Button
                                type="primary"
                                className="tableButton padding-all"
                                onClick={this.closeSucceedHint}
                            >
                                OK
                            </Button>
                        </Row>
                    </Row>
                </Modal>

            </Row>
        );
    }
}

export default Resubmit;
