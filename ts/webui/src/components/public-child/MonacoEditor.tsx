import * as React from 'react';
import { Spinner } from '@fluentui/react';
import MonacoEditor from 'react-monaco-editor';

interface MonacoEditorProps {
    content: string;
    loading: boolean;
    height: number;
}

class MonacoHTML extends React.Component<MonacoEditorProps, {}> {
    constructor(props: MonacoEditorProps) {
        super(props);
    }

    render(): React.ReactNode {
        const { content, loading, height } = this.props;
        return (
            <React.Fragment>
                {loading ? (
                    <Spinner
                        label='Wait, wait...'
                        ariaLive='assertive'
                        labelPosition='right'
                        styles={{ root: { width: '100%', height: height } }}
                    >
                        <MonacoEditor
                            width='100%'
                            height={height}
                            language='json'
                            value={content}
                            options={{
                                minimap: { enabled: false },
                                readOnly: true,
                                automaticLayout: true,
                                wordWrap: 'on'
                            }}
                        />
                    </Spinner>
                ) : (
                    <MonacoEditor
                        width='100%'
                        height={height}
                        language='json'
                        value={content}
                        options={{
                            minimap: { enabled: false },
                            readOnly: true,
                            automaticLayout: true,
                            wordWrap: 'on'
                        }}
                    />
                )}
            </React.Fragment>
        );
    }
}

export default MonacoHTML;
