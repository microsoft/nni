import * as React from 'react';
import axios from 'axios';
import { Row, Col, Input, Modal, Form, Button, Icon } from 'antd';
import { MANAGER_IP } from '../../static/const';
import { EXPERIMENT } from '../../static/datamodel';
import { FormComponentProps } from 'antd/lib/form';
const FormItem = Form.Item;
import './customized.scss';

interface CustomizeProps extends FormComponentProps {
    hyperParameter: string;
    visible: boolean;
    closeCustomizeModal: () => void;
}

interface CustomizeState {
    isShowSubmitSucceed: boolean;
    isShowSubmitFailed: boolean;
    isShowWarning: boolean;
    searchSpace: object;
    customParameters: object; // customized trial
}

class Customize extends React.Component<CustomizeProps, CustomizeState> {

    public _isCustomizeMount: boolean;
    constructor(props: CustomizeProps) {
        super(props);
        this.state = {
            isShowSubmitSucceed: false,
            isShowSubmitFailed: false,
            isShowWarning: false,
            searchSpace: EXPERIMENT.searchSpace,
            customParameters: {}
        };
    }

    // [submit click] user add a new trial [submit a trial]
    addNewTrial = () => {
        const { searchSpace } = this.state;
        const { hyperParameter } = this.props;
        const originParameter = JSON.parse(hyperParameter);
        // get user edited hyperParameter, ps: will change data type if you modify the input val
        const customized = this.props.form.getFieldsValue();
        // true: parameters are wrong
        let flag = false;
        Object.keys(customized).map(item => {
            if (item !== 'tag') {
                // unified data type
                if (typeof originParameter[item] === 'number' && typeof customized[item] === 'string') {
                    customized[item] = JSON.parse(customized[item]);
                }
                if (searchSpace[item]._type === 'choice') {
                    if (searchSpace[item]._value.find((val: string | number) =>
                        val === customized[item]) === undefined) {
                        flag = true;
                        return;
                    }
                } else {
                    if (customized[item] < searchSpace[item]._value[0]
                        || customized[item] > searchSpace[item]._value[1]) {
                        flag = true;
                        return;
                    }
                }
            }
        });
        if (flag !== false) {
            // open the warning modal
            if (this._isCustomizeMount === true) {
                this.setState(() => ({ isShowWarning: true, customParameters: customized }));
            }
        } else {
            // submit a customized job
            this.submitCustomize(customized);
        }

    }

    warningConfirm = () => {
        if (this._isCustomizeMount === true) {
            this.setState(() => ({ isShowWarning: false }));
        }
        const { customParameters } = this.state;
        this.submitCustomize(customParameters);
    }

    warningCancel = () => {
        if (this._isCustomizeMount === true) {
            this.setState(() => ({ isShowWarning: false }));
        }
    }

    submitCustomize = (customized: Object) => {
        // delete `tag` key
        for (let i in customized) {
            if (i === 'tag') {
                delete customized[i];
            }
        }
        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            data: customized
        })
            .then(res => {
                if (res.status === 200) {
                    if (this._isCustomizeMount === true) {
                        this.setState(() => ({ isShowSubmitSucceed: true }));
                        this.props.closeCustomizeModal();
                    }
                } else {
                    if (this._isCustomizeMount === true) {
                        this.setState(() => ({ isShowSubmitFailed: true }));
                    }
                }
            })
            .catch(error => {
                if (this._isCustomizeMount === true) {
                    this.setState(() => ({ isShowSubmitFailed: true }));
                }
            });
    }

    closeSucceedHint = () => {
        // also close customized trial modal
        if (this._isCustomizeMount === true) {
            this.setState(() => ({ isShowSubmitSucceed: false }));
            this.props.closeCustomizeModal();
        }
    }

    closeFailedHint = () => {
        // also close customized trial modal
        if (this._isCustomizeMount === true) {
            this.setState(() => ({ isShowSubmitFailed: false }));
            this.props.closeCustomizeModal();
        }
    }

    componentDidMount() {
        this._isCustomizeMount = true;
    }

    componentWillUnmount() {
        this._isCustomizeMount = false;
    }

    render() {
        const { visible, closeCustomizeModal, hyperParameter } = this.props;
        const { isShowSubmitSucceed, isShowSubmitFailed, isShowWarning } = this.state;
        const {
            form: { getFieldDecorator },
            // form: { getFieldDecorator, getFieldValue },
        } = this.props;
        const parameters = (hyperParameter !== '') ? JSON.parse(hyperParameter) : '';
        const warning = 'The parameters you set are not in our search space, this may cause the tuner to crash, Are'
            + ' you sure you want to continue submitting?';
        return (
            <Row>
                {/* form: search space */}
                <Modal
                    title="Customized trial setting"
                    visible={visible}
                    onCancel={closeCustomizeModal}
                    footer={null}
                    destroyOnClose={true}
                    maskClosable={false}
                    centered={true}
                >
                    {/* search space form */}
                    <Row className="hyper-box">
                        <Form>
                            {
                                Object.keys(parameters).map(item => {
                                    return (
                                        <Row key={item} className="hyper-form">
                                            <Col span={9} className="title">{item}</Col>
                                            <Col span={15} className="inputs">
                                                <FormItem key={item} style={{ marginBottom: 0 }}>
                                                    {getFieldDecorator(item, {
                                                        initialValue: parameters[item],
                                                    })(
                                                        <Input />
                                                    )}
                                                </FormItem>
                                            </Col>
                                        </Row>
                                    );
                                })
                            }
                            <Row key="tag" className="hyper-form tag-input">
                                <Col span={9} className="title">Tag</Col>
                                <Col span={15} className="inputs">
                                    <FormItem key="tag" style={{ marginBottom: 0 }}>
                                        {getFieldDecorator('tag', {
                                            initialValue: 'Customized',
                                        })(
                                            <Input />
                                        )}
                                    </FormItem>
                                </Col>
                            </Row>
                        </Form>
                    </Row>
                    <Row className="modal-button">
                        <Button
                            type="primary"
                            className="tableButton distance"
                            onClick={this.addNewTrial}
                        >
                            Submit
                        </Button>
                        <Button
                            className="tableButton cancelSty"
                            onClick={this.props.closeCustomizeModal}
                        >
                            Cancel
                        </Button>
                    </Row>
                    {/* control button */}
                </Modal>
                {/* clone: prompt succeed or failed */}
                <Modal
                    visible={isShowSubmitSucceed}
                    footer={null}
                    destroyOnClose={true}
                    maskClosable={false}
                    closable={false}
                    centered={true}
                >
                    <Row className="resubmit">
                        <Row>
                            <h2 className="title">
                                <span>
                                    <Icon type="check-circle" className="color-succ" />
                                    <b>Submit successfully</b>
                                </span>
                            </h2>
                            <div className="hint">
                                {/* don't return trial ID */}
                                {/* <span>You can find your customized trial by trial num.</span> */}
                                <span>Submit successfully.</span>
                            </div>
                        </Row>
                        <Row className="modal-button">
                            <Button
                                className="tableButton cancelSty"
                                onClick={this.closeSucceedHint}
                            >
                                OK
                            </Button>
                        </Row>
                    </Row>
                </Modal>
                <Modal
                    visible={isShowSubmitFailed}
                    footer={null}
                    destroyOnClose={true}
                    maskClosable={false}
                    closable={false}
                    centered={true}
                >
                    <Row className="resubmit">
                        <Row>
                            <h2 className="title">
                                <span>
                                    <Icon type="check-circle" className="color-error" />Submit Failed
                                </span>
                            </h2>
                            <div className="hint">
                                <span>Unknown error.</span>
                            </div>
                        </Row>
                        <Row className="modal-button">
                            <Button
                                className="tableButton cancelSty"
                                onClick={this.closeFailedHint}
                            >
                                OK
                            </Button>
                        </Row>
                    </Row>
                </Modal>
                {/* hyperParameter not match search space, warning modal */}
                <Modal
                    visible={isShowWarning}
                    footer={null}
                    destroyOnClose={true}
                    maskClosable={false}
                    closable={false}
                    centered={true}
                >
                    <Row className="resubmit">
                        <Row>
                            <h2 className="title">
                                <span>
                                    <Icon className="color-warn" type="warning" />Warning
                                </span>
                            </h2>
                            <div className="hint">
                                <span>{warning}</span>
                            </div>
                        </Row>
                        <Row className="modal-button center">
                            <Button
                                className="tableButton cancelSty distance"
                                onClick={this.warningConfirm}
                            >
                                Confirm
                            </Button>
                            <Button
                                className="tableButton cancelSty"
                                onClick={this.warningCancel}
                            >
                                Cancel
                            </Button>
                        </Row>
                    </Row>
                </Modal>

            </Row>

        );
    }
}

export default Form.create<FormComponentProps>()(Customize);