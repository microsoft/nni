import * as React from 'react';
import { Spin } from 'antd';
import { DRAWEROPTION } from '../../static/const';
import MonacoEditor from 'react-monaco-editor';

interface MonacoEditorProps {
    content: string;
    loading: boolean;
    height: number;
}

class MonacoHTML extends React.Component<MonacoEditorProps, {}> {

    public _isMonacoMount: boolean;

    constructor(props: MonacoEditorProps) {
        super(props);
    }

    componentDidMount(): void {
        this._isMonacoMount = true;
    }

    componentWillUnmount(): void {
        this._isMonacoMount = false;
    }

    render(): React.ReactNode{
        const { content, loading, height } = this.props;
        return (
            <div className="just-for-log">
                <Spin
                    // tip="Loading..."
                    style={{ width: '100%', height: height }}
                    spinning={loading}
                >
                    <MonacoEditor
                        width="100%"
                        height={height}
                        language="json"
                        value={content}
                        options={DRAWEROPTION}
                    />
                </Spin>
            </div>
        );
    }
}

export default MonacoHTML;
